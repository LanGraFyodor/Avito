from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from bge_pair_pilot import ap10, truths


DATA_DIR = Path("candidate_data")
PROJECTOR = Path("models/jina-reranker-v3-gguf/projector.safetensors")
FP32_CACHE = Path("cache/jina_listwise_first5.npz")
GGUF_CACHE = Path("cache/jina_listwise_q8_first5.npz")
SERVER = "http://127.0.0.1:8089"
TOP_K = 40
N_VALIDATION = 5
SPECIAL = {"query_embed_token": "<|rerank_token|>", "doc_embed_token": "<|embed_token|>"}


def format_prompt(query: str, docs: list[str]) -> str:
    for token in SPECIAL.values():
        query = query.replace(token, "")
    docs = [doc.replace(SPECIAL["query_embed_token"], "").replace(SPECIAL["doc_embed_token"], "") for doc in docs]
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    body = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    body += "\n".join(
        f'<passage id="{i}">\n{doc}{SPECIAL["doc_embed_token"]}\n</passage>'
        for i, doc in enumerate(docs)
    )
    body += f"\n<query>\n{query}{SPECIAL['query_embed_token']}\n</query>"
    return prefix + body + suffix


def post(path: str, payload: dict, timeout: float = 600.0):
    request = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {details}") from error


def load_projector():
    with safe_open(str(PROJECTOR), framework="pt", device="cpu") as handle:
        w0 = handle.get_tensor("projector.0.weight").float().numpy()
        w2 = handle.get_tensor("projector.2.weight").float().numpy()
    return w0, w2


def embed_prompt(prompt: str, w0: np.ndarray, w2: np.ndarray):
    token_ids = np.asarray(post("/tokenize", {"content": prompt, "add_special": True})["tokens"])
    payload = post("/embedding", {"content": prompt})
    if isinstance(payload, list):
        if len(payload) != 1:
            raise RuntimeError(f"unexpected embedding response items={len(payload)}")
        payload = payload[0]
    hidden = np.asarray(payload["embedding"], dtype=np.float32)
    if len(hidden) != len(token_ids):
        raise RuntimeError(f"token/embedding mismatch: {len(token_ids)} != {len(hidden)}")
    doc_positions = np.flatnonzero(token_ids == 151670)
    query_positions = np.flatnonzero(token_ids == 151671)
    if len(doc_positions) < 1 or len(query_positions) != 1:
        raise RuntimeError(
            f"special token mismatch docs={len(doc_positions)} queries={len(query_positions)}"
        )
    selected = np.vstack((hidden[doc_positions], hidden[query_positions[0]]))
    projected = np.maximum(selected @ w0.T, 0.0) @ w2.T
    return projected[:-1], projected[-1]


def score_prompt(prompt: str, w0: np.ndarray, w2: np.ndarray) -> np.ndarray:
    docs, query = embed_prompt(prompt, w0, w2)
    return (docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query) + 1e-12)).astype(np.float32)


def lexical_similarity(texts, train, validation):
    normalized = [value.lower().replace("ё", "е") for value in texts]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(normalized)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=160_000
    ).fit_transform(normalized)
    return (
        0.45 * (word[validation] @ word[train].T).toarray()
        + 0.55 * (char[validation] @ char[train].T).toarray()
    ).astype(np.float32)


def aggregate(logits, candidates, train_rows, labels, article_to_col, n_articles, temperature):
    output = np.zeros((len(logits), n_articles), dtype=np.float32)
    weights = np.exp((logits - logits.max(axis=1, keepdims=True)) / temperature)
    for row in range(len(logits)):
        for candidate, weight in zip(candidates[row], weights[row]):
            source = int(train_rows[int(candidate)])
            for article_id in labels[source]:
                output[row, article_to_col[article_id]] += float(weight)
    return output


def evaluate(scores, article_ids, actual):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return float(np.mean([ap10(article_ids[p], set(a)) for p, a in zip(order, actual)]))


def main():
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = truths(calibration.ground_truth)
    groups = np.asarray([" ".join(map(str, sorted(row))) for row in labels])
    train_rows, validation_rows = next(
        GroupKFold(5).split(np.arange(len(calibration)), groups=groups)
    )
    validation_rows = validation_rows[:N_VALIDATION]
    texts = calibration.query_text.astype(str).tolist()
    lexical = lexical_similarity(texts, train_rows, validation_rows)
    candidates = np.argsort(-lexical, axis=1, kind="stable")[:, :TOP_K]
    if GGUF_CACHE.exists():
        logits = np.load(GGUF_CACHE)["logits"]
    else:
        w0, w2 = load_projector()
        rows = []
        started = time.time()
        for local_row, validation_row in enumerate(validation_rows):
            docs = [texts[int(train_rows[int(position)])] for position in candidates[local_row]]
            prompt = format_prompt(texts[int(validation_row)], docs)
            rows.append(score_prompt(prompt, w0, w2))
            print(
                f"gguf={local_row + 1}/{len(validation_rows)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        logits = np.asarray(rows, dtype=np.float32)
        GGUF_CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(GGUF_CACHE, logits=logits)

    fp32 = np.load(FP32_CACHE)["logits"]
    pearson = float(np.corrcoef(fp32.reshape(-1), logits.reshape(-1))[0, 1])
    rank_fp32 = np.argsort(np.argsort(fp32.reshape(-1)))
    rank_gguf = np.argsort(np.argsort(logits.reshape(-1)))
    spearman = float(np.corrcoef(rank_fp32, rank_gguf)[0, 1])
    top1 = float(np.mean(np.argmax(fp32, axis=1) == np.argmax(logits, axis=1)))
    top5 = float(
        np.mean(
            [len(set(np.argsort(-left)[:5]) & set(np.argsort(-right)[:5])) / 5 for left, right in zip(fp32, logits)]
        )
    )
    print(
        f"q8_vs_fp32 pearson={pearson:.6f} spearman={spearman:.6f} "
        f"top1={top1:.4f} top5={top5:.4f}"
    )
    actual = [labels[int(row)] for row in validation_rows]
    for temperature in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50):
        scores = aggregate(
            logits, candidates, train_rows, labels, article_to_col, len(article_ids), temperature
        )
        print(f"q8_temperature={temperature:.2f} map={evaluate(scores, article_ids, actual):.6f}")


if __name__ == "__main__":
    main()
