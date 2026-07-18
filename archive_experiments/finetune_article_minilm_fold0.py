from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
from lxml import html as lxml_html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bge_pair_pilot import ap10, truths
from solution import normalize_text


DATA_DIR = Path("candidate_data")
BASE_MODEL = Path("models/mmarco-minilm")
CACHE_DIR = Path("cache/minilm_article")
SEED = 2466955
TOP_K = 100


def ensure_base_model():
    """Download only the open-source model weights needed for local training."""
    if (BASE_MODEL / "config.json").exists():
        return
    BASE_MODEL.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        local_dir=BASE_MODEL,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )


def article_windows(articles):
    chunks, chunk_cols = [], []
    for col, row in enumerate(articles.itertuples()):
        title = normalize_text(row.title)
        try:
            root = lxml_html.fromstring(str(row.body))
            for tag in root.xpath("//script|//style|//noscript|//svg"):
                tag.drop_tree()
            visible = normalize_text(" ".join(root.itertext()))
        except (ValueError, lxml_html.ParserError):
            visible = normalize_text(str(row.body))
        words = visible.split()
        local = []
        for start in range(0, max(1, len(words)), 60):
            text = normalize_text(f"{title}. {' '.join(words[start : start + 100])}")
            if text and text not in local:
                local.append(text)
            if len(local) >= 20:
                break
        if not local:
            local = [title]
        chunks.extend(local)
        chunk_cols.extend([col] * len(local))
    return chunks, np.asarray(chunk_cols, dtype=np.int64)


def best_chunks(chunks, chunk_cols, queries, n_articles):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=220_000,
        token_pattern=r"(?u)\b[\w-]{2,}\b", dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(chunks + queries)
    similarity = (matrix[len(chunks) :] @ matrix[: len(chunks)].T).toarray().astype(np.float32)
    output = np.zeros((len(queries), n_articles), dtype=np.int64)
    for col in range(n_articles):
        indices = np.flatnonzero(chunk_cols == col)
        local = similarity[:, indices]
        output[:, col] = indices[np.argmax(local, axis=1)]
    return output


def rank_score(scores):
    order = np.argsort(-scores, axis=1, kind="stable")
    output = np.empty_like(scores, dtype=np.float32)
    values = 1.0 / np.log2(np.arange(2, scores.shape[1] + 2, dtype=np.float32))
    output[np.arange(len(scores))[:, None], order] = values
    return output


def evaluate(scores, article_ids, labels, rows):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return float(np.mean([
        ap10(article_ids[prediction], set(labels[int(row)]))
        for prediction, row in zip(order, rows)
    ]))


def make_training_pairs(train_rows, labels, article_to_col, direct, best, chunks, queries):
    pair_queries, pair_docs, targets = [], [], []
    for row_value in train_rows:
        row = int(row_value)
        positives = [article_to_col[value] for value in labels[row]]
        positive_set = set(positives)
        negatives = [
            int(col) for col in np.argsort(-direct[row], kind="stable")
            if int(col) not in positive_set
        ][:18]
        # Repeat positives so BCE does not drown them among hard negatives.
        for col in positives:
            for _ in range(3):
                pair_queries.append(queries[row])
                pair_docs.append(chunks[best[row, col]])
                targets.append(1.0)
        for col in negatives:
            pair_queries.append(queries[row])
            pair_docs.append(chunks[best[row, col]])
            targets.append(0.0)
    return pair_queries, pair_docs, np.asarray(targets, dtype=np.float32)


def train_model(tokenizer, queries, documents, targets, output_model):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, local_files_only=True
    )
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[:8]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    encoded = tokenizer(
        queries, documents, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt",
    )
    target_tensor = torch.from_numpy(targets)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.5e-5, weight_decay=0.02,
    )
    model.train()
    batch_size = 24
    order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(SEED))
    losses, started = [], time.time()
    for step, start in enumerate(range(0, len(order), batch_size)):
        indices = order[start : start + batch_size]
        batch = {key: value[indices] for key, value in encoded.items()}
        logits = model(**batch).logits.reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tensor[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 40 == 0:
            print(
                f"train_step={step}/{(len(order) + batch_size - 1) // batch_size} "
                f"loss={losses[-1]:.6f} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    print(f"train_loss={np.mean(losses):.6f} elapsed={time.time() - started:.1f}s", flush=True)
    output_model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_model)
    tokenizer.save_pretrained(output_model)
    return model.eval()


@torch.inference_mode()
def score_validation(model, tokenizer, validation_rows, candidates, best, chunks, queries):
    pair_queries, pair_docs = [], []
    for local, row_value in enumerate(validation_rows):
        row = int(row_value)
        for col in candidates[local]:
            pair_queries.append(queries[row])
            pair_docs.append(chunks[best[row, int(col)]])
    values = []
    for start in range(0, len(pair_queries), 96):
        encoded = tokenizer(
            pair_queries[start : start + 96], pair_docs[start : start + 96],
            padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        values.append(model(**encoded).logits.reshape(-1).float().numpy())
        if start % 960 == 0:
            print(f"scored={min(start + 96, len(pair_queries))}/{len(pair_queries)}", flush=True)
    return np.concatenate(values).reshape(len(validation_rows), TOP_K).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    fold = args.fold
    output_model = Path(f"models/mmarco-minilm-avito-article-group{fold}")
    score_path = CACHE_DIR / f"group{fold}_scores.npz"
    ensure_base_model()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = truths(calibration.ground_truth)
    groups = np.asarray([" ".join(map(str, sorted(row))) for row in labels])
    train_rows, validation_rows = list(
        GroupKFold(5).split(np.arange(len(calibration)), groups=groups)
    )[fold]
    queries = [normalize_text(value) for value in calibration.query_text.astype(str)]
    corpus_path = CACHE_DIR / "clean_corpus.npz"
    best_path = CACHE_DIR / "best_chunks.npy"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if corpus_path.exists():
        payload = np.load(corpus_path)
        chunks = payload["chunks"].astype(str).tolist()
        chunk_cols = payload["chunk_cols"]
    else:
        chunks, chunk_cols = article_windows(articles)
        np.savez_compressed(
            corpus_path, chunks=np.asarray(chunks, dtype=str), chunk_cols=chunk_cols
        )
    print(f"articles={len(articles)} clean_chunks={len(chunks)}", flush=True)
    if best_path.exists():
        best = np.load(best_path)
    else:
        best = best_chunks(chunks, chunk_cols, queries, len(articles))
        np.save(best_path, best)

    retrieval = np.load("cache/ltr/retrieval.npz")
    component_weights = (1.0, 0.55, 0.55, 0.60, 0.65, 0.65)
    direct = sum(
        weight * retrieval[f"cal_{index}"]
        for index, weight in enumerate(component_weights)
    )
    candidates = np.argsort(-direct[validation_rows], axis=1, kind="stable")[:, :TOP_K]
    recall = []
    for local, row_value in enumerate(validation_rows):
        actual = set(labels[int(row_value)])
        predicted = set(map(int, article_ids[candidates[local]]))
        recall.append(len(actual & predicted) / len(actual))
    print(f"candidate_recall={np.mean(recall):.6f}", flush=True)

    if score_path.exists():
        score_payload = np.load(score_path)
        scores = score_payload["scores"]
        value = float(score_payload["map"][0])
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
        if output_model.joinpath("config.json").exists():
            model = AutoModelForSequenceClassification.from_pretrained(
                output_model, local_files_only=True
            ).eval()
        else:
            pair_queries, pair_docs, targets = make_training_pairs(
                train_rows, labels, article_to_col, direct, best, chunks, queries
            )
            print(
                f"train_pairs={len(targets)} positives={(targets > 0).mean():.4f}",
                flush=True,
            )
            model = train_model(tokenizer, pair_queries, pair_docs, targets, output_model)

        logits = score_validation(
            model, tokenizer, validation_rows, candidates, best, chunks, queries
        )
        scores = np.full((len(validation_rows), len(article_ids)), -100.0, dtype=np.float32)
        scores[np.arange(len(validation_rows))[:, None], candidates] = logits
        value = evaluate(scores, article_ids, labels, validation_rows)
        np.savez_compressed(
            score_path, scores=scores,
            validation_rows=validation_rows, article_ids=article_ids,
            candidates=candidates, map=np.asarray([value]),
        )
    print(f"article_crossencoder_map={value:.6f}", flush=True)

    jina_path = Path(f"cache/jina_q8/group{fold}_scores.npz")
    if jina_path.exists():
        jina = np.load(jina_path)["scores"]
        jina_rank = rank_score(jina)
        article_rank = rank_score(scores)
        for weight in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
            fused = jina_rank + weight * article_rank
            print(
                f"jina_plus_article={weight:.2f} "
                f"map={evaluate(fused, article_ids, labels, validation_rows):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
