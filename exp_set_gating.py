from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import exp_set_decoder as decoder


def main() -> None:
    articles = pd.read_feather(decoder.DATA_DIR / "articles.f")
    calibration = pd.read_feather(decoder.DATA_DIR / "calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = decoder.labels_from_frame(calibration, article_to_col)
    scores = np.load(decoder.CACHE_DIR / "standard_oof.npz")["scores"]
    order, ranks = decoder.rank_matrix(scores)
    splits = list(
        KFold(5, shuffle=True, random_state=decoder.SEED).split(np.arange(len(calibration)))
    )
    raw_graph = decoder.html_links(articles, article_to_col)
    states = decoder.build_fold_states(splits, labels, len(articles), raw_graph)

    base_predictions = order[:, :10]
    modified_predictions = decoder.set_prediction(
        order,
        ranks,
        states,
        temperature=6.0,
        size_power=0.5,
        prior=0.0,
        anchor_n=1,
        min_count=1,
        keep=3,
        margin_gate=0.0,
        evidence_gate=0.0,
    )
    base_ap = decoder.row_ap(base_predictions, labels)
    modified_ap = decoder.row_ap(modified_predictions, labels)
    changed = np.any(modified_predictions != base_predictions, axis=1)

    add_rank = np.full(len(calibration), np.inf, dtype=np.float64)
    conditional = np.zeros(len(calibration), dtype=np.float64)
    anchor_frequency = np.zeros(len(calibration), dtype=np.float64)
    normalized_gap = np.zeros(len(calibration), dtype=np.float64)
    linked = np.zeros(len(calibration), dtype=bool)
    for state in states:
        for row in state.validation:
            anchor = int(order[row, 0])
            additions = [
                int(value)
                for value in modified_predictions[row, 3:]
                if int(value) not in set(map(int, order[row, :3]))
            ]
            if changed[row] and additions:
                addition = additions[0]
                add_rank[row] = ranks[row, addition]
                conditional[row] = state.conditional[anchor, addition]
                anchor_frequency[row] = state.frequency[anchor]
                linked[row] = bool(raw_graph[anchor, addition] or raw_graph[addition, anchor])
            local = scores[row, order[row, :30]]
            normalized_gap[row] = (local[0] - local[1]) / (np.std(local) + 1e-6)

    configs = []
    ap_rows = []
    # No-op is deliberately included.  If a tuning subset does not support a
    # correction, nested selection can safely retain the base ranking.
    configs.append({"no_op": True})
    ap_rows.append(base_ap)
    for max_rank, min_conditional, min_frequency, min_gap, max_gap, link_policy in itertools.product(
        (4, 5, 6, 8, 20),
        (0.0, 0.03, 0.05, 0.08, 0.12),
        (1, 5, 10, 20, 40),
        (0.0, 0.15, 0.30),
        (0.50, 1.00, math.inf),
        ("any", "unlinked"),
    ):
        gate = (
            changed
            & (add_rank <= max_rank)
            & (conditional >= min_conditional)
            & (anchor_frequency >= min_frequency)
            & (normalized_gap >= min_gap)
            & (normalized_gap <= max_gap)
        )
        if link_policy == "unlinked":
            gate &= ~linked
        configs.append(
            {
                "max_rank": max_rank,
                "min_conditional": min_conditional,
                "min_frequency": min_frequency,
                "min_gap": min_gap,
                "max_gap": "inf" if np.isinf(max_gap) else max_gap,
                "link_policy": link_policy,
                "rows": int(gate.sum()),
            }
        )
        ap_rows.append(np.where(gate, modified_ap, base_ap))

    ap_matrix = np.vstack(ap_rows)
    base_folds = decoder.fold_values(base_ap, splits)

    records = []
    for index, (config, values) in enumerate(zip(configs, ap_matrix)):
        folds = decoder.fold_values(values, splits)
        deltas = [value - base for value, base in zip(folds, base_folds)]
        records.append(
            {
                "candidate": index,
                "params": config,
                "map": float(values.mean()),
                "delta": float(values.mean() - base_ap.mean()),
                "fold_map": folds,
                "fold_delta": deltas,
                "min_fold_delta": float(min(deltas)),
            }
        )

    nested = np.empty(len(calibration), dtype=np.float64)
    selected = []
    for outer, (_, validation) in enumerate(splits):
        tuning_folds = [fold for fold in range(5) if fold != outer]
        tuning_rows = np.concatenate([splits[fold][1] for fold in tuning_folds])
        tuning_mean = ap_matrix[:, tuning_rows].mean(axis=1)
        tuning_worst = np.asarray(
            [min(record["fold_delta"][fold] for fold in tuning_folds) for record in records]
        )
        # Penalising a bad tuning fold sharply protects the final test transform
        # from an accidental gain concentrated in one intent cluster.
        objective = tuning_mean + 0.35 * tuning_worst
        best = int(np.argmax(objective))
        nested[validation] = ap_matrix[best, validation]
        selected.append({"outer_fold": outer, **records[best]})

    by_mean = sorted(records, key=lambda record: record["map"], reverse=True)
    by_robust = sorted(
        records,
        key=lambda record: (record["min_fold_delta"], record["map"]),
        reverse=True,
    )
    payload = {
        "baseline": {"map": float(base_ap.mean()), "fold_map": base_folds},
        "ungated": {
            "map": float(modified_ap.mean()),
            "delta": float(modified_ap.mean() - base_ap.mean()),
            "changed": int(changed.sum()),
            "wins": int((modified_ap > base_ap).sum()),
            "losses": int((modified_ap < base_ap).sum()),
        },
        "nested": {
            "map": float(nested.mean()),
            "delta": float(nested.mean() - base_ap.mean()),
            "fold_map": decoder.fold_values(nested, splits),
            "selected": selected,
        },
        "candidate_count": len(records),
        "top_by_mean": by_mean[:30],
        "top_by_worst_fold": by_robust[:30],
    }
    Path("exp_set_gating_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        "exp_set_gating_search.npz",
        ap=ap_matrix,
        base_ap=base_ap,
        modified_ap=modified_ap,
        changed=changed,
        add_rank=add_rank,
        conditional=conditional,
        anchor_frequency=anchor_frequency,
        normalized_gap=normalized_gap,
        linked=linked,
        nested_ap=nested,
    )
    print(
        f"baseline={base_ap.mean():.9f} ungated={modified_ap.mean():.9f} "
        f"nested={nested.mean():.9f}",
        flush=True,
    )
    for record in by_mean[:15]:
        print(
            f"candidate={record['candidate']} map={record['map']:.9f} "
            f"delta={record['delta']:+.9f} min_fold={record['min_fold_delta']:+.9f} "
            f"fold_delta={[round(x, 6) for x in record['fold_delta']]} "
            f"params={record['params']}",
            flush=True,
        )


if __name__ == "__main__":
    import math

    main()
