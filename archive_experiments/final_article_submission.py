from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune_article_minilm_fold0 import best_chunks
from solution import normalize_text


DATA_DIR = Path("candidate_data")
CACHE_DIR = Path("cache/minilm_article")
MODEL_DIR = Path("models/mmarco-minilm-avito-article-group0")
BASELINE = Path("best_public_061.csv")
OUTPUT = Path("answer.csv")
TOP_RETRIEVAL = 100
TOP_ARTICLE = 30
ARTICLE_WEIGHT = 0.75


def sparse_rank(rows, article_ids, width):
    id_to_col = {int(value): col for col, value in enumerate(article_ids)}
    output = np.zeros((len(rows), len(article_ids)), dtype=np.float32)
    weights = 1.0 / np.log2(np.arange(2, width + 2, dtype=np.float32))
    for row, values in enumerate(rows):
        columns = [id_to_col[int(value)] for value in values[:width]]
        output[row, columns] = weights[: len(columns)]
    return output


@torch.inference_mode()
def score_test(model, tokenizer, queries, candidates, chunks, best):
    path = CACHE_DIR / "test_group0_article_logits.npz"
    logits = np.full((len(queries), TOP_RETRIEVAL), np.nan, dtype=np.float32)
    if path.exists():
        old = np.load(path)
        if np.array_equal(old["candidates"], candidates):
            logits = old["logits"]
    block_size = 10
    for block_start in range(0, len(queries), block_size):
        block_rows = np.arange(block_start, min(block_start + block_size, len(queries)))
        if np.isfinite(logits[block_rows]).all():
            continue
        pair_queries, documents = [], []
        for row in block_rows:
            pair_queries.extend([queries[int(row)]] * TOP_RETRIEVAL)
            documents.extend(
                chunks[best[int(row), int(col)]] for col in candidates[int(row)]
            )
        values = []
        for start in range(0, len(documents), 96):
            encoded = tokenizer(
                pair_queries[start : start + 96],
                documents[start : start + 96],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            values.append(model(**encoded).logits.reshape(-1).float().numpy())
        logits[block_rows] = np.concatenate(values).reshape(len(block_rows), TOP_RETRIEVAL)
        np.savez_compressed(path, logits=logits, candidates=candidates)
        print(
            f"test_article_scored={int(block_rows[-1]) + 1}/{len(queries)}",
            flush=True,
        )
    return logits


def validate(frame, test, valid_ids):
    assert frame.columns.tolist() == ["query_id", "answer"]
    assert frame.query_id.tolist() == test.query_id.tolist()
    for value in frame.answer:
        ids = list(map(int, value.split()))
        assert len(ids) == 10 and len(set(ids)) == 10 and set(ids) <= valid_ids


def main():
    torch.set_num_threads(8)
    articles = pd.read_feather(DATA_DIR / "articles.f")
    test = pd.read_feather(DATA_DIR / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    baseline = pd.read_csv(BASELINE)
    queries = [normalize_text(value) for value in test.query_text.astype(str)]

    corpus = np.load(CACHE_DIR / "clean_corpus.npz")
    chunks = corpus["chunks"].astype(str).tolist()
    chunk_cols = corpus["chunk_cols"]
    best_path = CACHE_DIR / "best_chunks_test.npy"
    if best_path.exists():
        best = np.load(best_path)
    else:
        best = best_chunks(chunks, chunk_cols, queries, len(articles))
        np.save(best_path, best)

    retrieval = np.load("cache/ltr/retrieval.npz")
    component_weights = (1.0, 0.55, 0.55, 0.60, 0.65, 0.65)
    direct = sum(
        weight * retrieval[f"test_{index}"]
        for index, weight in enumerate(component_weights)
    )
    candidates = np.argsort(-direct, axis=1, kind="stable")[:, :TOP_RETRIEVAL]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).eval()
    logits = score_test(model, tokenizer, queries, candidates, chunks, best)
    local_order = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_ARTICLE]
    article_rows = [article_ids[candidates[row, local_order[row]]].tolist() for row in range(len(test))]
    baseline_rows = [list(map(int, value.split())) for value in baseline.answer.astype(str)]
    score = sparse_rank(baseline_rows, article_ids, 10)
    score += ARTICLE_WEIGHT * sparse_rank(article_rows, article_ids, TOP_ARTICLE)
    order = np.argsort(-score, axis=1, kind="stable")[:, :10]
    result = test[["query_id"]].copy()
    result["answer"] = [" ".join(map(str, article_ids[row])) for row in order]
    validate(result, test, set(map(int, article_ids)))
    result.to_csv(OUTPUT, index=False, lineterminator="\n")

    old = baseline_rows
    new = [list(map(int, value.split())) for value in result.answer]
    top1_changed = np.mean([left[0] != right[0] for left, right in zip(old, new)])
    overlap = np.mean([len(set(left) & set(right)) for left, right in zip(old, new)])
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(
        f"saved={OUTPUT} rows=500 top1_changed={top1_changed:.3f} "
        f"mean_top10_overlap={overlap:.3f} sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
