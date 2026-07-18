"""Closed-set intent / multilabel experiments for the Avito retrieval task.

This file deliberately produces only OOF diagnostics.  It never writes a test
submission.  The experiment uses the 79 article labels observed in calibration
and the same five shuffled folds as ``ltr_solution.py``.

The strongest previously computed independent multilabel channel is loaded from
``cache/supervised_multilabel_scores.npz`` when available.  In addition, this
script computes label-wise and label-powerset kernel-ridge predictions directly
from word and character TF-IDF query kernels.  All channels are evaluated alone
and as rank ensembles with the LTR OOF matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold


DATA_DIR = Path("candidate_data")
LTR_OOF = Path("cache/ltr/standard_oof.npz")
SUPERVISED_CACHE = Path("cache/supervised_multilabel_scores.npz")
OUTPUT = Path("exp_intent_best_oof.npz")
SEED = 2_466_955


def normalize_text(value: object) -> str:
    import html
    import re

    text = html.unescape(str(value)).lower().replace("ё", "е")
    text = text.replace("<money>", " деньги ").replace("<date>", " дата ")
    text = re.sub(r"[^0-9a-zа-я<>]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def parse_truth(values: pd.Series) -> list[list[int]]:
    return [[int(item) for item in str(value).split()] for value in values]


def ap10(prediction: np.ndarray, truth: set[int]) -> float:
    hits = 0
    value = 0.0
    for rank, article_id in enumerate(prediction[:10], 1):
        if int(article_id) in truth:
            hits += 1
            value += hits / rank
    return value / min(len(truth), 10)


@dataclass
class Data:
    article_ids: np.ndarray
    compact_ids: np.ndarray
    compact_cols: np.ndarray
    labels: list[list[int]]
    y: np.ndarray
    calibration_texts: list[str]
    all_texts: list[str]


def load_data() -> Data:
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    test = pd.read_feather(DATA_DIR / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    labels = parse_truth(calibration.ground_truth)
    compact_ids = np.asarray(sorted({value for row in labels for value in row}), dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    compact_cols = np.asarray([article_to_col[int(value)] for value in compact_ids], dtype=np.int64)
    label_to_col = {int(value): col for col, value in enumerate(compact_ids)}
    y = np.zeros((len(calibration), len(compact_ids)), dtype=np.float32)
    for row, values in enumerate(labels):
        for value in values:
            y[row, label_to_col[int(value)]] = 1.0
    calibration_texts = [normalize_text(value) for value in calibration.query_text]
    all_texts = calibration_texts + [normalize_text(value) for value in test.query_text]
    return Data(
        article_ids=article_ids,
        compact_ids=compact_ids,
        compact_cols=compact_cols,
        labels=labels,
        y=y,
        calibration_texts=calibration_texts,
        all_texts=all_texts,
    )


def evaluate_compact(scores: np.ndarray, data: Data, rows: np.ndarray | None = None) -> float:
    if rows is None:
        rows = np.arange(len(data.labels))
    order = np.argsort(-scores[rows], axis=1, kind="stable")[:, :10]
    return float(
        np.mean(
            [
                ap10(data.compact_ids[prediction], set(data.labels[int(row)]))
                for prediction, row in zip(order, rows)
            ]
        )
    )


def rank01(scores: np.ndarray) -> np.ndarray:
    """Convert each row to stable [0, 1] descending rank scores."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(scores, dtype=np.float32)
    scale = np.linspace(1.0, 0.0, scores.shape[1], endpoint=False, dtype=np.float32)
    ranks[np.arange(len(scores))[:, None], order] = scale
    return ranks


def query_kernel(texts: list[str], n_calibration: int) -> np.ndarray:
    word = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=120_000, dtype=np.float32
    ).fit_transform(texts)[:n_calibration]
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=180_000,
        dtype=np.float32,
    ).fit_transform(texts)[:n_calibration]
    return (0.45 * (word @ word.T) + 0.55 * (char @ char.T)).toarray().astype(np.float64)


def solve_kernel_ridge(
    kernel: np.ndarray,
    targets: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> np.ndarray:
    prediction = np.zeros((len(kernel), targets.shape[1]), dtype=np.float32)
    for train, validation in splits:
        system = kernel[np.ix_(train, train)].copy()
        system.flat[:: len(train) + 1] += alpha
        factor = cho_factor(system, lower=True, check_finite=False)
        coefficients = cho_solve(factor, targets[train], check_finite=False)
        prediction[validation] = kernel[np.ix_(validation, train)] @ coefficients
    return prediction


def powerset_probabilities(
    kernel: np.ndarray,
    data: Data,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    temperature: float,
) -> np.ndarray:
    keys = [tuple(np.flatnonzero(row)) for row in data.y]
    prediction = np.zeros_like(data.y, dtype=np.float32)
    for train, validation in splits:
        train_keys = sorted({keys[int(row)] for row in train})
        key_to_col = {key: col for col, key in enumerate(train_keys)}
        target = np.zeros((len(train), len(train_keys)), dtype=np.float64)
        for local_row, global_row in enumerate(train):
            target[local_row, key_to_col[keys[int(global_row)]]] = 1.0
        system = kernel[np.ix_(train, train)].copy()
        system.flat[:: len(train) + 1] += alpha
        factor = cho_factor(system, lower=True, check_finite=False)
        coefficients = cho_solve(factor, target, check_finite=False)
        logits = kernel[np.ix_(validation, train)] @ coefficients
        logits = logits / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        class_to_labels = np.zeros((len(train_keys), data.y.shape[1]), dtype=np.float64)
        for col, key in enumerate(train_keys):
            class_to_labels[col, list(key)] = 1.0
        prediction[validation] = probability @ class_to_labels
    return prediction


def fold_values(scores: np.ndarray, data: Data, splits) -> list[float]:
    return [evaluate_compact(scores, data, validation) for _, validation in splits]


def full_matrix(compact_scores: np.ndarray, data: Data) -> np.ndarray:
    result = np.full((len(compact_scores), len(data.article_ids)), -100.0, dtype=np.float32)
    result[:, data.compact_cols] = compact_scores
    return result


def main() -> None:
    data = load_data()
    splits = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(data.labels))))
    baseline_full = np.load(LTR_OOF)["scores"].astype(np.float32)
    baseline = baseline_full[:, data.compact_cols]
    channels: dict[str, np.ndarray] = {"ltr_seen": rank01(baseline)}
    print(f"ltr_raw={evaluate_compact(baseline, data):.6f}")
    print(f"ltr_seen={evaluate_compact(channels['ltr_seen'], data):.6f}")

    if SUPERVISED_CACHE.exists():
        cached = np.load(SUPERVISED_CACHE)
        svm = cached["standard_svm"][:, data.compact_cols].astype(np.float32)
        channels["multilabel_svm"] = rank01(svm)
        print(f"multilabel_svm={evaluate_compact(svm, data):.6f}")

    kernel = query_kernel(data.all_texts, len(data.labels))
    for alpha in (0.5, 1.0, 2.0):
        score = solve_kernel_ridge(kernel, data.y.astype(np.float64), splits, alpha)
        channels[f"label_krr_{alpha:g}"] = rank01(score)
        print(f"label_krr_{alpha:g}={evaluate_compact(score, data):.6f}")
    for alpha in (0.5, 1.0, 2.0):
        for temperature in (0.10, 0.20, 0.35):
            score = powerset_probabilities(kernel, data, splits, alpha, temperature)
            name = f"powerset_krr_{alpha:g}_t{temperature:g}"
            channels[name] = rank01(score)
            print(f"{name}={evaluate_compact(score, data):.6f}")

    base = channels["ltr_seen"]
    candidates: list[tuple[float, str, float, np.ndarray]] = [
        (evaluate_compact(base, data), "ltr_seen", 0.0, base)
    ]
    for name, channel in channels.items():
        if name == "ltr_seen":
            continue
        weights = sorted(set(np.arange(0.10, 1.01, 0.05).round(2).tolist() + [0.67, 0.68]))
        for weight in weights:
            ensemble = base + float(weight) * channel
            candidates.append((evaluate_compact(ensemble, data), name, float(weight), ensemble))
    # Confidence gate: use the independent classifier only when both models
    # substantially agree on the intent.  This avoids damaging the difficult
    # tail where the closed-set classifier is extrapolating.  The threshold is
    # discrete and the useful weight range is deliberately broad (0.7--0.9).
    if "multilabel_svm" in channels:
        ltr_order = np.argsort(-baseline, axis=1, kind="stable")[:, :3]
        svm_raw = np.load(SUPERVISED_CACHE)["standard_svm"][:, data.compact_cols]
        svm_order = np.argsort(-svm_raw, axis=1, kind="stable")[:, :3]
        agreement = np.asarray(
            [len(set(left) & set(right)) >= 2 for left, right in zip(ltr_order, svm_order)],
            dtype=np.float32,
        )
        for weight in (0.70, 0.80, 0.90):
            ensemble = base + weight * agreement[:, None] * channels["multilabel_svm"]
            candidates.append(
                (evaluate_compact(ensemble, data), "multilabel_svm_gate_top3_ge2", weight, ensemble)
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    for value, name, weight, score in candidates[:15]:
        folds = ",".join(f"{item:.6f}" for item in fold_values(score, data, splits))
        print(f"ensemble={value:.6f} channel={name} weight={weight:.2f} folds=[{folds}]")

    best_value, best_name, best_weight, best_score = candidates[0]
    np.savez_compressed(
        OUTPUT,
        scores=full_matrix(best_score, data),
        compact_scores=best_score.astype(np.float32),
        compact_article_ids=data.compact_ids,
        map=np.asarray([best_value], dtype=np.float64),
        channel=np.asarray([best_name]),
        weight=np.asarray([best_weight], dtype=np.float64),
        fold_maps=np.asarray(fold_values(best_score, data, splits), dtype=np.float64),
    )
    print(f"saved={OUTPUT} best={best_value:.6f} channel={best_name} weight={best_weight:.2f}")


if __name__ == "__main__":
    main()
