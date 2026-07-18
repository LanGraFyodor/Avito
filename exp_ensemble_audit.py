"""Fold-safe audit of conservative ensembles for the Avito ranking task.

This experiment never writes a submission.  It treats ``standard_oof`` as the
anchor and tests only rank-based rules that can also be applied to a frozen
public top-10.  All reported labels for a validation row come from models that
did not train on that row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 2466955
TOP_ALT = 40
WEIGHTS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)


def ap10(prediction: np.ndarray, actual: set[int]) -> float:
    hits = 0
    total = 0.0
    for rank, value in enumerate(prediction[:10], 1):
        if int(value) in actual:
            hits += 1
            total += hits / rank
    return total / min(len(actual), 10)


def average_precision_rows(
    predictions: np.ndarray, article_ids: np.ndarray, truths: list[set[int]]
) -> np.ndarray:
    return np.asarray(
        [ap10(article_ids[row], actual) for row, actual in zip(predictions, truths)],
        dtype=np.float64,
    )


def fuse_orders(
    anchor: np.ndarray,
    alternative: np.ndarray,
    weight: float,
    mode: str,
    alternative_width: int = TOP_ALT,
) -> np.ndarray:
    """Rank-fuse two already sorted column arrays without a 793-wide sort."""
    output = np.empty((len(anchor), 10), dtype=np.int64)
    anchor_value = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    alt_value = 1.0 / np.log2(
        np.arange(2, alternative_width + 2, dtype=np.float64)
    )
    for row, (anchor_row, alt_row) in enumerate(zip(anchor, alternative)):
        locked: list[int] = []
        if mode == "reorder":
            candidates = anchor_row[:10].tolist()
        elif mode == "reorder_keep1":
            locked = anchor_row[:1].tolist()
            candidates = anchor_row[1:10].tolist()
        elif mode == "reorder_keep2":
            locked = anchor_row[:2].tolist()
            candidates = anchor_row[2:10].tolist()
        elif mode == "replace":
            candidates = list(
                dict.fromkeys(
                    np.r_[anchor_row[:10], alt_row[:alternative_width]].tolist()
                )
            )
        elif mode == "anchor40":
            candidates = list(
                dict.fromkeys(
                    np.r_[anchor_row[:40], alt_row[:alternative_width]].tolist()
                )
            )
        else:
            raise ValueError(mode)
        anchor_scores = {
            int(column): float(anchor_value[rank])
            for rank, column in enumerate(anchor_row[:10])
        }
        alt_scores = {
            int(column): float(alt_value[rank])
            for rank, column in enumerate(alt_row[:alternative_width])
        }
        values = np.asarray(
            [
                anchor_scores.get(int(column), 0.0)
                + weight * alt_scores.get(int(column), 0.0)
                for column in candidates
            ]
        )
        order = np.argsort(-values, kind="stable")[:10]
        ranked = np.asarray(candidates, dtype=np.int64)[order]
        output[row] = np.r_[np.asarray(locked, dtype=np.int64), ranked]
    return output


def bootstrap(delta: np.ndarray, label_sets: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    iid_index = rng.integers(0, len(delta), size=(10_000, len(delta)))
    iid = delta[iid_index].mean(axis=1)
    unique = np.unique(label_sets)
    members = [np.flatnonzero(label_sets == value) for value in unique]
    group_sum = np.asarray([delta[rows].sum() for rows in members])
    group_count = np.asarray([len(rows) for rows in members])
    sampled = rng.integers(0, len(members), size=(10_000, len(members)))
    clustered = group_sum[sampled].sum(axis=1) / group_count[sampled].sum(axis=1)
    return {
        "iid_ci95": np.quantile(iid, (0.025, 0.5, 0.975)).tolist(),
        "iid_probability_gain": float(np.mean(iid > 0)),
        "label_set_cluster_ci95": np.quantile(
            clustered, (0.025, 0.5, 0.975)
        ).tolist(),
        "label_set_cluster_probability_gain": float(np.mean(clustered > 0)),
    }


def main() -> None:
    print("load data", flush=True)
    articles = pd.read_feather("candidate_data/articles.f")
    calibration = pd.read_feather("candidate_data/calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    truths = [set(map(int, str(value).split())) for value in calibration.ground_truth]
    label_sets = np.asarray(
        [" ".join(map(str, sorted(values))) for values in truths]
    )

    print("load score caches", flush=True)
    anchor_scores = np.load("cache/ltr/standard_oof.npz")["scores"]
    supervised = np.load("cache/supervised_multilabel_scores.npz")
    alternative_scores = supervised["standard_svm"]
    print("sort cached scores", flush=True)
    anchor_order = np.argsort(
        -anchor_scores, axis=1, kind="stable"
    )[:, :TOP_ALT]
    alternative_order = np.argsort(
        -alternative_scores, axis=1, kind="stable"
    )[:, :TOP_ALT]

    fold_id = np.empty(len(calibration), dtype=np.int64)
    splits = list(
        KFold(5, shuffle=True, random_state=SEED).split(
            np.arange(len(calibration))
        )
    )
    for fold, (_, validation) in enumerate(splits):
        fold_id[validation] = fold

    anchor_ap = average_precision_rows(anchor_order[:, :10], article_ids, truths)
    overlap3 = np.asarray(
        [
            len(set(left[:3]) & set(right[:3]))
            for left, right in zip(anchor_order, alternative_order)
        ]
    )
    gates = {
        "all": np.ones(len(calibration), dtype=bool),
        "overlap3_not_1": overlap3 != 1,
        "overlap3_at_least_2": overlap3 >= 2,
        "top1_disagrees": anchor_order[:, 0] != alternative_order[:, 0],
    }

    candidates: list[dict[str, object]] = []
    prediction_cache: dict[tuple[str, float], np.ndarray] = {}
    ap_cache: dict[tuple[str, float], np.ndarray] = {}
    for mode in ("reorder", "reorder_keep1", "reorder_keep2", "replace", "anchor40"):
        print(f"grid mode={mode}", flush=True)
        for weight in WEIGHTS:
            predictions = fuse_orders(
                anchor_order, alternative_order, weight, mode
            )
            prediction_cache[(mode, weight)] = predictions
            candidate_ap = average_precision_rows(predictions, article_ids, truths)
            ap_cache[(mode, weight)] = candidate_ap
            for gate_name, gate in gates.items():
                delta = np.where(gate, candidate_ap - anchor_ap, 0.0)
                candidates.append(
                    {
                        "mode": mode,
                        "weight": weight,
                        "gate": gate_name,
                        "map": float(anchor_ap.mean() + delta.mean()),
                        "gain": float(delta.mean()),
                        "fold_gains": [
                            float(delta[fold_id == fold].mean())
                            for fold in range(5)
                        ],
                        "changed_rows": int(
                            np.sum(gate & np.any(predictions != anchor_order[:, :10], axis=1))
                        ),
                    }
                )

    print("LOFO meta-selection", flush=True)
    # Hyperparameter-selection audit: choose on four complete folds, score on the fifth.
    transferable = [row for row in candidates if row["mode"] != "anchor40"]
    meta_ap = anchor_ap.copy()
    meta_choices: list[dict[str, object]] = []
    for fold in range(5):
        train = fold_id != fold
        validation = fold_id == fold
        best = max(
            transferable,
            key=lambda row: np.mean(
                np.where(
                    gates[str(row["gate"])][train],
                    ap_cache[(str(row["mode"]), float(row["weight"]))][train]
                    - anchor_ap[train],
                    0.0,
                )
            ),
        )
        predictions = prediction_cache[(str(best["mode"]), float(best["weight"]))]
        candidate_ap = ap_cache[(str(best["mode"]), float(best["weight"]))][validation]
        gate = gates[str(best["gate"])][validation]
        meta_ap[validation] = np.where(gate, candidate_ap, anchor_ap[validation])
        meta_choices.append(
            {
                "fold": fold,
                "mode": best["mode"],
                "weight": best["weight"],
                "gate": best["gate"],
                "heldout_gain": float(
                    (meta_ap[validation] - anchor_ap[validation]).mean()
                ),
            }
        )

    robust = [row for row in transferable if min(row["fold_gains"]) > 0]
    best_robust = max(robust, key=lambda row: float(row["gain"]))
    robust_predictions = prediction_cache[
        (str(best_robust["mode"]), float(best_robust["weight"]))
    ]
    robust_gate = gates[str(best_robust["gate"])]
    robust_ap = np.where(
        robust_gate,
        average_precision_rows(robust_predictions, article_ids, truths),
        anchor_ap,
    )
    recommended = next(
        row
        for row in candidates
        if row["mode"] == "reorder"
        and row["weight"] == 0.9
        and row["gate"] == "overlap3_not_1"
    )
    recommended_predictions = prediction_cache[("reorder", 0.9)]
    recommended_gate = gates["overlap3_not_1"]
    recommended_ap = np.where(
        recommended_gate,
        ap_cache[("reorder", 0.9)],
        anchor_ap,
    )

    # Apply only for change-count diagnostics.  No submission is written.
    public = pd.read_csv("best_public_061.csv")
    public_ids = np.asarray(
        [list(map(int, str(value).split())) for value in public.answer],
        dtype=np.int64,
    )
    article_to_col = {int(value): column for column, value in enumerate(article_ids)}
    public_columns = np.asarray(
        [[article_to_col[value] for value in row] for row in public_ids],
        dtype=np.int64,
    )
    test_alternative = supervised["test_svm"]
    test_alternative_order = np.argsort(
        -test_alternative, axis=1, kind="stable"
    )[:, :TOP_ALT]
    test_overlap3 = np.asarray(
        [
            len(set(left[:3]) & set(right[:3]))
            for left, right in zip(public_columns, test_alternative_order)
        ]
    )
    test_gate = {
        "all": np.ones(len(public), dtype=bool),
        "overlap3_not_1": test_overlap3 != 1,
        "overlap3_at_least_2": test_overlap3 >= 2,
        "top1_disagrees": public_columns[:, 0] != test_alternative_order[:, 0],
    }[str(recommended["gate"])]
    padded_public = np.pad(
        public_columns, ((0, 0), (0, TOP_ALT - 10)), constant_values=-1
    )
    test_predictions = fuse_orders(
        padded_public,
        test_alternative_order,
        float(recommended["weight"]),
        str(recommended["mode"]),
    )
    applied_test = np.where(test_gate[:, None], test_predictions, public_columns)

    print("bootstrap and report", flush=True)
    report = {
        "anchor_map": float(anchor_ap.mean()),
        "standard_svm_map": float(
            average_precision_rows(
                alternative_order[:, :10], article_ids, truths
            ).mean()
        ),
        "best_robust_transferable": best_robust,
        "best_robust_bootstrap": bootstrap(robust_ap - anchor_ap, label_sets),
        "recommended_conservative": recommended,
        "recommended_conservative_bootstrap": bootstrap(
            recommended_ap - anchor_ap, label_sets
        ),
        "lofo_meta": {
            "map": float(meta_ap.mean()),
            "gain": float((meta_ap - anchor_ap).mean()),
            "fold_gains": [
                float((meta_ap - anchor_ap)[fold_id == fold].mean())
                for fold in range(5)
            ],
            "choices": meta_choices,
            "bootstrap": bootstrap(meta_ap - anchor_ap, label_sets),
        },
        "test_change_diagnostics": {
            "gate_rows": int(test_gate.sum()),
            "changed_rows": int(
                np.sum(np.any(applied_test != public_columns, axis=1))
            ),
            "changed_top1": int(
                np.sum(applied_test[:, 0] != public_columns[:, 0])
            ),
            "mean_top10_overlap": float(
                np.mean(
                    [
                        len(set(left) & set(right))
                        for left, right in zip(applied_test, public_columns)
                    ]
                )
            ),
        },
        "all_candidates": sorted(
            candidates, key=lambda row: float(row["gain"]), reverse=True
        ),
    }
    Path("exp_ensemble_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        "exp_ensemble_oof.npz",
        anchor_ap=anchor_ap,
        robust_ap=robust_ap,
        recommended_ap=recommended_ap,
        lofo_meta_ap=meta_ap,
        fold_id=fold_id,
        overlap3=overlap3,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
