from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import exp_set_decoder as decoder


def main() -> None:
    articles = pd.read_feather(decoder.DATA_DIR / "articles.f")
    calibration = pd.read_feather(decoder.DATA_DIR / "calibration.f")
    test = pd.read_feather(decoder.DATA_DIR / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = decoder.labels_from_frame(calibration, article_to_col)

    payload = np.load(decoder.CACHE_DIR / "test_scores.npz")
    scores = payload["scores"]
    if "article_ids" in payload.files and not np.array_equal(payload["article_ids"], article_ids):
        raise RuntimeError("Article order in test_scores.npz does not match articles.f")
    order, ranks = decoder.rank_matrix(scores)
    graph = decoder.html_links(articles, article_to_col)
    # The transform is fitted once on all calibration labels only.  Test labels
    # are never accessed and query_id is not used as a feature.
    states = decoder.build_fold_states(
        [(np.arange(len(calibration)), np.arange(len(test)))],
        labels,
        len(articles),
        graph,
    )
    state = states[0]
    set_modified = decoder.set_prediction(
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
    base = order[:, :10]
    changed_by_set = np.any(set_modified != base, axis=1)
    gate = np.zeros(len(test), dtype=bool)
    diagnostics = []
    for row in range(len(test)):
        if not changed_by_set[row]:
            continue
        prefix = set(map(int, order[row, :3]))
        additions = [int(value) for value in set_modified[row, 3:] if int(value) not in prefix]
        if not additions:
            continue
        anchor = int(order[row, 0])
        addition = additions[0]
        add_rank = int(ranks[row, addition])
        conditional = float(state.conditional[anchor, addition])
        frequency = float(state.frequency[anchor])
        accepted = add_rank <= 5 and conditional >= 0.03 and frequency >= 10
        gate[row] = accepted
        diagnostics.append(
            {
                "row": row,
                "query_id": int(test.query_id.iloc[row]),
                "anchor": int(article_ids[anchor]),
                "addition": int(article_ids[addition]),
                "base_rank_zero_based": add_rank,
                "p_addition_given_anchor": conditional,
                "anchor_frequency": frequency,
                "accepted": bool(accepted),
            }
        )

    transformed = base.copy()
    transformed[gate] = set_modified[gate]
    transformed_ids = article_ids[transformed]
    if transformed_ids.shape != (len(test), 10):
        raise RuntimeError(f"Unexpected result shape: {transformed_ids.shape}")
    if any(len(set(map(int, row))) != 10 for row in transformed_ids):
        raise RuntimeError("Duplicate article_id in transformed top-10")

    np.savez_compressed(
        "exp_set_test_candidate.npz",
        query_id=test.query_id.to_numpy(),
        article_ids=transformed_ids,
        base_article_ids=article_ids[base],
        accepted=gate,
    )
    Path("exp_set_test_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"test_rows={len(test)} set_changed={int(changed_by_set.sum())} "
        f"accepted={int(gate.sum())} output=exp_set_test_candidate.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
