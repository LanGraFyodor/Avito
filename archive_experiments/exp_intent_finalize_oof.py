"""Materialize the best closed-set OOF rank ensemble (no test output)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


WEIGHT = 0.80
SEED = 2_466_955
OUTPUT = Path("exp_intent_best_oof.npz")


def rank01(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(values, dtype=np.float32)
    ranks[np.arange(len(values))[:, None], order] = np.linspace(
        1.0, 0.0, values.shape[1], endpoint=False, dtype=np.float32
    )
    return ranks, order


def ap10(prediction, truth):
    hits = 0
    value = 0.0
    for rank, article_id in enumerate(prediction[:10], 1):
        if int(article_id) in truth:
            hits += 1
            value += hits / rank
    return value / min(len(truth), 10)


def main() -> None:
    articles = pd.read_feather("candidate_data/articles.f")
    calibration = pd.read_feather("candidate_data/calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    labels = [[int(item) for item in str(value).split()] for value in calibration.ground_truth]
    compact_ids = np.asarray(sorted({item for row in labels for item in row}), dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    compact_cols = np.asarray([article_to_col[int(value)] for value in compact_ids])

    ltr = np.load("cache/ltr/standard_oof.npz")["scores"][:, compact_cols]
    svm = np.load("cache/supervised_multilabel_scores.npz")["standard_svm"][:, compact_cols]
    ltr_rank, ltr_order = rank01(ltr)
    svm_rank, svm_order = rank01(svm)
    gate = np.asarray(
        [len(set(left[:3]) & set(right[:3])) >= 2 for left, right in zip(ltr_order, svm_order)],
        dtype=np.float32,
    )
    compact_scores = ltr_rank + WEIGHT * gate[:, None] * svm_rank
    full_scores = np.full((len(calibration), len(article_ids)), -100.0, dtype=np.float32)
    full_scores[:, compact_cols] = compact_scores

    order = np.argsort(-compact_scores, axis=1, kind="stable")[:, :10]
    row_ap = np.asarray(
        [ap10(compact_ids[prediction], set(truth)) for prediction, truth in zip(order, labels)]
    )
    splits = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(labels))))
    fold_maps = np.asarray([row_ap[validation].mean() for _, validation in splits])
    np.savez_compressed(
        OUTPUT,
        scores=full_scores,
        compact_scores=compact_scores,
        compact_article_ids=compact_ids,
        map=np.asarray([row_ap.mean()]),
        fold_maps=fold_maps,
        channel=np.asarray(["multilabel_svm_gate_top3_ge2"]),
        weight=np.asarray([WEIGHT]),
        gate=gate.astype(np.int8),
    )
    print(f"saved={OUTPUT} map={row_ap.mean():.6f} folds={fold_maps.tolist()}")


if __name__ == "__main__":
    main()
