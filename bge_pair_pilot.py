from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("candidate_data")
MODEL_DIR = Path("models/bge-reranker-v2-m3")
CACHE = Path("cache/bge_pair_group0.npz")


def truths(values) -> list[list[int]]:
    return [list(map(int, str(value).split())) for value in values]


def ap10(prediction: np.ndarray, actual: set[int]) -> float:
    hits = 0
    total = 0.0
    for rank, value in enumerate(prediction[:10], 1):
        if int(value) in actual:
            hits += 1
            total += hits / rank
    return total / min(len(actual), 10)


def evaluate(scores: np.ndarray, article_ids: np.ndarray, actual: list[list[int]]) -> float:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return float(
        np.mean(
            [ap10(article_ids[row], set(labels)) for row, labels in zip(order, actual)]
        )
    )


def lexical_similarity(texts: list[str], train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    normalized = [value.lower().replace("ё", "е") for value in texts]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(normalized)
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=160_000,
    ).fit_transform(normalized)
    return (
        0.45 * (word[validation] @ word[train].T).toarray()
        + 0.55 * (char[validation] @ char[train].T).toarray()
    ).astype(np.float32)


@torch.inference_mode()
def score_pairs(queries: list[str], documents: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    print("quantizing linear layers to int8", flush=True)
    model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    output: list[np.ndarray] = []
    for start in range(0, len(queries), 64):
        encoded = tokenizer(
            queries[start : start + 64],
            documents[start : start + 64],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        logits = model(**encoded).logits.reshape(-1).float().cpu().numpy()
        output.append(logits)
        if start % 640 == 0:
            print(f"scored={min(start + 64, len(queries))}/{len(queries)}", flush=True)
    return np.concatenate(output).astype(np.float32)


def aggregate(
    values: np.ndarray,
    candidate_positions: np.ndarray,
    train_rows: np.ndarray,
    train_truths: list[list[int]],
    article_to_col: dict[int, int],
    n_articles: int,
    mode: str,
) -> np.ndarray:
    output = np.zeros((len(values), n_articles), dtype=np.float32)
    for query_row in range(len(values)):
        local_values = values[query_row]
        if mode == "rank":
            local_order = np.argsort(-local_values, kind="stable")
            local_values = 1.0 / np.log2(np.arange(2, len(local_order) + 2))
        else:
            local_order = np.arange(len(local_values))
            local_values = local_values - local_values.min()
            local_values = np.exp(local_values - local_values.max())
        for rank_position, weight in zip(local_order, local_values):
            source_row = int(train_rows[candidate_positions[query_row, rank_position]])
            for article_id in train_truths[source_row]:
                col = article_to_col[article_id]
                output[query_row, col] += float(weight)
    maximum = output.max(axis=1, keepdims=True)
    return np.divide(output, maximum, out=np.zeros_like(output), where=maximum > 0)


def main() -> None:
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    all_truths = truths(calibration.ground_truth)
    groups = np.asarray([" ".join(map(str, sorted(row))) for row in all_truths])
    train_rows, validation_rows = next(
        GroupKFold(5).split(np.arange(len(calibration)), groups=groups)
    )
    texts = calibration.query_text.astype(str).tolist()
    lexical = lexical_similarity(texts, train_rows, validation_rows)
    candidate_count = 40
    candidates = np.argsort(-lexical, axis=1, kind="stable")[:, :candidate_count]
    validation_truths = [all_truths[int(row)] for row in validation_rows]

    pair_queries: list[str] = []
    pair_documents: list[str] = []
    for local_row, validation_row in enumerate(validation_rows):
        for position in candidates[local_row]:
            pair_queries.append(texts[int(validation_row)])
            pair_documents.append(texts[int(train_rows[position])])
    print(
        f"train={len(train_rows)} validation={len(validation_rows)} pairs={len(pair_queries)}",
        flush=True,
    )
    if CACHE.exists():
        pair_scores = np.load(CACHE)["scores"]
    else:
        pair_scores = score_pairs(pair_queries, pair_documents)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, scores=pair_scores)
    pair_scores = pair_scores.reshape(len(validation_rows), candidate_count)

    lexical_values = np.take_along_axis(lexical, candidates, axis=1)
    lexical_scores = aggregate(
        lexical_values,
        candidates,
        train_rows,
        all_truths,
        article_to_col,
        len(article_ids),
        "softmax",
    )
    bge_rank = aggregate(
        pair_scores,
        candidates,
        train_rows,
        all_truths,
        article_to_col,
        len(article_ids),
        "rank",
    )
    bge_softmax = aggregate(
        pair_scores,
        candidates,
        train_rows,
        all_truths,
        article_to_col,
        len(article_ids),
        "softmax",
    )
    for name, score in (
        ("lexical", lexical_scores),
        ("bge_rank", bge_rank),
        ("bge_softmax", bge_softmax),
    ):
        print(name, evaluate(score, article_ids, validation_truths), flush=True)
    for weight in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
        score = lexical_scores + weight * bge_rank
        print(
            f"blend_rank={weight:.2f} map={evaluate(score, article_ids, validation_truths):.6f}",
            flush=True,
        )
    for weight in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
        score = lexical_scores + weight * bge_softmax
        print(
            f"blend_softmax={weight:.2f} map={evaluate(score, article_ids, validation_truths):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
