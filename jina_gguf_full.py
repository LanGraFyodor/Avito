from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, KFold

from bge_pair_pilot import ap10, truths
from jina_gguf_pilot import format_prompt, load_projector, score_prompt


DATA_DIR = Path("candidate_data")
CACHE_DIR = Path("cache/jina_q8")
PILOT_CACHE = Path("cache/jina_listwise_q8_first5.npz")
TOP_K = 40


def fit_lexical(texts):
    normalized = [value.lower().replace("ё", "е") for value in texts]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(normalized)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=180_000
    ).fit_transform(normalized)
    return word, char


def similarities(word, char, target, source):
    return (
        0.45 * (word[target] @ word[source].T).toarray()
        + 0.55 * (char[target] @ char[source].T).toarray()
    ).astype(np.float32)


def aggregate(logits, candidates, source_rows, labels, article_to_col, n_articles, temperature):
    output = np.zeros((len(logits), n_articles), dtype=np.float32)
    weights = np.exp((logits - logits.max(axis=1, keepdims=True)) / temperature)
    for row in range(len(logits)):
        for candidate, weight in zip(candidates[row], weights[row]):
            source = int(source_rows[int(candidate)])
            for article_id in labels[source]:
                output[row, article_to_col[article_id]] += float(weight)
    return output


def evaluate(scores, article_ids, labels, rows):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return float(
        np.mean(
            [ap10(article_ids[prediction], set(labels[int(row)])) for prediction, row in zip(order, rows)]
        )
    )


def score_job(name, texts, target_rows, source_rows, candidates):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.npz"
    logits = np.full((len(target_rows), TOP_K), np.nan, dtype=np.float32)
    if path.exists():
        old = np.load(path)
        if np.array_equal(old["target_rows"], target_rows) and np.array_equal(
            old["candidates"], candidates
        ):
            logits = old["logits"]
    elif name == "group0" and PILOT_CACHE.exists():
        pilot = np.load(PILOT_CACHE)["logits"]
        logits[: len(pilot)] = pilot

    w0, w2 = load_projector()
    started = time.time()
    for local_row, target_row in enumerate(target_rows):
        if np.isfinite(logits[local_row]).all():
            continue
        documents = [texts[int(source_rows[int(position)])] for position in candidates[local_row]]
        logits[local_row] = score_prompt(
            format_prompt(texts[int(target_row)], documents), w0, w2
        )
        np.savez_compressed(
            path,
            logits=logits,
            candidates=candidates,
            target_rows=target_rows,
            source_rows=source_rows,
        )
        print(
            f"{name}={local_row + 1}/{len(target_rows)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        choices=(
            "group0", "group1", "group2", "group3", "group4",
            "standard0", "standard1", "standard2", "standard3", "standard4",
            "test",
        ),
        required=True,
    )
    args = parser.parse_args()
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    test = pd.read_feather(DATA_DIR / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = truths(calibration.ground_truth)
    calibration_texts = calibration.query_text.astype(str).tolist()
    if args.job.startswith("group") or args.job.startswith("standard"):
        texts = calibration_texts
        word, char = fit_lexical(texts)
        if args.job.startswith("group"):
            groups = np.asarray([" ".join(map(str, sorted(row))) for row in labels])
            fold = int(args.job.removeprefix("group"))
            source_rows, target_rows = list(
                GroupKFold(5).split(np.arange(len(calibration)), groups=groups)
            )[fold]
        else:
            fold = int(args.job.removeprefix("standard"))
            source_rows, target_rows = list(
                KFold(5, shuffle=True, random_state=2466955).split(
                    np.arange(len(calibration))
                )
            )[fold]
    else:
        texts = calibration_texts + test.query_text.astype(str).tolist()
        word, char = fit_lexical(texts)
        source_rows = np.arange(len(calibration), dtype=np.int64)
        target_rows = np.arange(len(calibration), len(texts), dtype=np.int64)
    lexical = similarities(word, char, target_rows, source_rows)
    candidates = np.argsort(-lexical, axis=1, kind="stable")[:, :TOP_K]
    logits = score_job(args.job, texts, target_rows, source_rows, candidates)

    best = (-1.0, None, None)
    if args.job.startswith("group") or args.job.startswith("standard"):
        for temperature in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.55):
            scores = aggregate(
                logits, candidates, source_rows, labels, article_to_col, len(article_ids), temperature
            )
            value = evaluate(scores, article_ids, labels, target_rows)
            print(f"temperature={temperature:.2f} {args.job}_map={value:.6f}")
            if value > best[0]:
                best = (value, temperature, scores)
        np.savez_compressed(
            CACHE_DIR / f"{args.job}_scores.npz",
            scores=best[2],
            target_rows=target_rows,
            article_ids=article_ids,
            map=np.asarray([best[0]], dtype=np.float32),
            temperature=np.asarray([best[1]], dtype=np.float32),
        )
        print(f"best_{args.job}={best[0]:.6f} temperature={best[1]}")
    else:
        # Temperature is selected on group0 and fixed before test prediction.
        temperature_path = CACHE_DIR / "group0_scores.npz"
        temperature = (
            float(np.load(temperature_path)["temperature"][0])
            if temperature_path.exists()
            else 0.20
        )
        scores = aggregate(
            logits, candidates, source_rows, labels, article_to_col, len(article_ids), temperature
        )
        np.savez_compressed(
            CACHE_DIR / "test_scores.npz", scores=scores, article_ids=article_ids
        )
        order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
        answer = test[["query_id"]].copy()
        answer["answer"] = [" ".join(map(str, article_ids[row])) for row in order]
        answer.to_csv("answer_jina_query.csv", index=False, lineterminator="\n")
        print("saved answer_jina_query.csv")


if __name__ == "__main__":
    main()
