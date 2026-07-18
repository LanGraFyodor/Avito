from __future__ import annotations

import argparse
import collections
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


DATA_DIR = Path("candidate_data")
CACHE_DIR = Path("cache/ltr")
SEED = 2466955


def labels_from_frame(frame: pd.DataFrame, article_to_col: dict[int, int]) -> list[set[int]]:
    return [
        {article_to_col[int(value)] for value in str(raw).split()}
        for raw in frame.ground_truth
    ]


def ap10(prediction: np.ndarray, actual: set[int]) -> float:
    hits = 0
    value = 0.0
    for rank, article in enumerate(prediction[:10], 1):
        if int(article) in actual:
            hits += 1
            value += hits / rank
    return value / min(len(actual), 10)


def row_ap(predictions: np.ndarray, labels: list[set[int]]) -> np.ndarray:
    return np.asarray([ap10(row, actual) for row, actual in zip(predictions, labels)])


def html_links(articles: pd.DataFrame, article_to_col: dict[int, int]) -> np.ndarray:
    graph = np.zeros((len(articles), len(articles)), dtype=np.float32)
    known_ids = set(article_to_col)
    for source, body in zip(articles.article_id, articles.body.astype(str)):
        source = int(source)
        for raw in re.findall(r"(?:support\.avito\.ru/)?articles/(\d+)", body, re.I):
            target = int(raw)
            if source in known_ids and target in known_ids and source != target:
                graph[article_to_col[source], article_to_col[target]] = 1.0
    return graph


@dataclass
class FoldState:
    train: np.ndarray
    validation: np.ndarray
    members: np.ndarray
    sizes: np.ndarray
    counts: np.ndarray
    conditional: np.ndarray
    frequency: np.ndarray
    html: np.ndarray


def build_fold_states(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: list[set[int]],
    n_articles: int,
    graph: np.ndarray,
) -> list[FoldState]:
    states = []
    for train, validation in splits:
        counter = collections.Counter(tuple(sorted(labels[int(row)])) for row in train)
        max_size = max(map(len, counter))
        members = np.full((len(counter), max_size), n_articles, dtype=np.int32)
        sizes = np.empty(len(counter), dtype=np.int32)
        counts = np.empty(len(counter), dtype=np.float32)
        for row, (key, count) in enumerate(counter.items()):
            members[row, : len(key)] = key
            sizes[row] = len(key)
            counts[row] = count

        # Labels are extremely sparse (one to four positives per query), so a
        # direct sparse count is both exact and much faster than dense Y.T @ Y.
        frequency = np.zeros(n_articles, dtype=np.float32)
        co = np.zeros((n_articles, n_articles), dtype=np.float32)
        for global_row in train:
            values = list(labels[int(global_row)])
            frequency[values] += 1.0
            for source in values:
                co[source, values] += 1.0
        conditional = np.divide(
            co,
            frequency[:, None],
            out=np.zeros_like(co),
            where=frequency[:, None] > 0,
        )
        np.fill_diagonal(conditional, 0.0)

        # Only links whose endpoints occurred as labels in this training fold are
        # allowed.  This makes the graph transform fold-safe.
        known = frequency > 0
        local_html = graph.copy()
        local_html[~known, :] = 0.0
        local_html[:, ~known] = 0.0
        # A reverse link is weaker evidence, but is still useful for navigation
        # articles that link to a more specific child article.
        local_html = np.maximum(local_html, 0.55 * local_html.T)
        np.fill_diagonal(local_html, 0.0)
        states.append(
            FoldState(
                train=train,
                validation=validation,
                members=members,
                sizes=sizes,
                counts=counts,
                conditional=conditional,
                frequency=frequency,
                html=local_html,
            )
        )
    return states


def rank_matrix(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1], dtype=np.int32)
    return order, ranks


def assemble(prefix: np.ndarray, additions: list[int], baseline: np.ndarray) -> np.ndarray:
    result: list[int] = []
    used: set[int] = set()
    for value in list(prefix) + additions + list(baseline):
        value = int(value)
        if value not in used:
            result.append(value)
            used.add(value)
        if len(result) == 10:
            break
    return np.asarray(result, dtype=np.int32)


def set_prediction(
    order: np.ndarray,
    ranks: np.ndarray,
    states: list[FoldState],
    temperature: float,
    size_power: float,
    prior: float,
    anchor_n: int,
    min_count: int,
    keep: int,
    margin_gate: float,
    evidence_gate: float,
) -> np.ndarray:
    predictions = order[:, :10].copy()
    n_articles = order.shape[1]
    for state in states:
        rows = state.validation
        safe_members = np.minimum(state.members, n_articles - 1)
        member_ranks = ranks[rows][:, safe_members]
        valid = state.members[None, :, :] < n_articles
        evidence = np.where(valid, np.exp(-member_ranks / temperature), 0.0).sum(axis=2)
        evidence /= state.sizes[None, :] ** size_power
        evidence += prior * np.log1p(state.counts)[None, :]
        contains_anchor = np.where(valid, member_ranks < anchor_n, False).any(axis=2)
        eligible = contains_anchor & (state.counts[None, :] >= min_count)
        evidence = np.where(eligible, evidence, -1e9)
        best = np.argmax(evidence, axis=1)
        top = evidence[np.arange(len(rows)), best]
        if evidence.shape[1] > 1:
            second = np.partition(evidence, -2, axis=1)[:, -2]
        else:
            second = np.full(len(rows), -1e9, dtype=np.float32)
        margin = top - second
        for local, global_row in enumerate(rows):
            if top[local] < evidence_gate or margin[local] < margin_gate:
                continue
            selected = state.members[best[local], : state.sizes[best[local]]]
            additions = sorted(
                (int(value) for value in selected if int(value) not in order[global_row, :keep]),
                key=lambda value: int(ranks[global_row, value]),
            )
            predictions[global_row] = assemble(
                order[global_row, :keep], additions, order[global_row]
            )
    return predictions


def graph_prediction(
    order: np.ndarray,
    states: list[FoldState],
    anchor_n: int,
    anchor_decay: float,
    html_weight: float,
    keep: int,
    additions_n: int,
    score_gate: float,
    min_frequency: int,
) -> np.ndarray:
    predictions = order[:, :10].copy()
    for state in states:
        for global_row in state.validation:
            anchors = order[global_row, :anchor_n]
            weights = anchor_decay ** np.arange(anchor_n, dtype=np.float32)
            score = weights @ state.conditional[anchors]
            if html_weight:
                score += html_weight * (weights @ state.html[anchors])
            score[state.frequency < min_frequency] = -1.0
            score[order[global_row, :keep]] = -1.0
            candidates = np.argsort(-score, kind="stable")[:additions_n]
            additions = [int(value) for value in candidates if score[value] >= score_gate]
            if additions:
                predictions[global_row] = assemble(
                    order[global_row, :keep], additions, order[global_row]
                )
    return predictions


def fold_values(values: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    return [float(values[validation].mean()) for _, validation in splits]


def candidate_record(
    family: str,
    params: dict,
    predictions: np.ndarray,
    labels: list[set[int]],
    base_ap: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, np.ndarray]:
    values = row_ap(predictions, labels)
    folds = fold_values(values, splits)
    base_folds = fold_values(base_ap, splits)
    deltas = [value - base for value, base in zip(folds, base_folds)]
    return (
        {
            "family": family,
            "params": params,
            "map": float(values.mean()),
            "delta": float(values.mean() - base_ap.mean()),
            "fold_map": folds,
            "fold_delta": deltas,
            "min_fold_delta": float(min(deltas)),
            "changed_rows": int(np.any(predictions != 0, axis=1).sum()),
        },
        values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="exp_set_report.json")
    parser.add_argument("--family", choices=("set", "graph", "both"), default="both")
    args = parser.parse_args()

    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = labels_from_frame(calibration, article_to_col)
    scores = np.load(CACHE_DIR / "standard_oof.npz")["scores"]
    order, ranks = rank_matrix(scores)
    splits = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(calibration))))
    graph = html_links(articles, article_to_col)
    states = build_fold_states(splits, labels, len(articles), graph)
    base_ap = row_ap(order[:, :10], labels)
    base_folds = fold_values(base_ap, splits)
    print(f"baseline={base_ap.mean():.9f} folds={base_folds}", flush=True)

    records: list[dict] = []
    ap_rows: list[np.ndarray] = []

    # Compact, pre-declared grid: it includes conservative and aggressive modes,
    # but is small enough to audit.  Hyperparameters are later selected using
    # leave-one-fold-out selection, never using the selected fold's labels.
    if args.family in ("set", "both"):
        for temperature in (6.0, 12.0):
            for size_power in (0.5, 1.0):
                for prior in (0.0, 0.20):
                    for anchor_n in (1,):
                        for min_count in (1,):
                            for keep in (1, 2, 3):
                                for margin_gate in (0.0,):
                                    params = {
                                        "temperature": temperature,
                                        "size_power": size_power,
                                        "prior": prior,
                                        "anchor_n": anchor_n,
                                        "min_count": min_count,
                                        "keep": keep,
                                        "margin_gate": margin_gate,
                                        "evidence_gate": 0.0,
                                    }
                                    prediction = set_prediction(order, ranks, states, **params)
                                    record, values = candidate_record(
                                        "label_set", params, prediction, labels, base_ap, splits
                                    )
                                    records.append(record)
                                    ap_rows.append(values)
        print(f"set_grid_done candidates={len(records)}", flush=True)

    if args.family in ("graph", "both"):
        for anchor_n in (1, 2):
            for anchor_decay in (0.70,):
                for html_weight in (0.0, 0.25, 0.60):
                    for keep in (1, 2):
                        for additions_n in (1, 2):
                            for score_gate in (0.50, 0.75):
                                for min_frequency in (1, 2):
                                    params = {
                                        "anchor_n": anchor_n,
                                        "anchor_decay": anchor_decay,
                                        "html_weight": html_weight,
                                        "keep": keep,
                                        "additions_n": additions_n,
                                        "score_gate": score_gate,
                                        "min_frequency": min_frequency,
                                    }
                                    prediction = graph_prediction(order, states, **params)
                                    record, values = candidate_record(
                                        "cooccurrence_graph", params, prediction, labels, base_ap, splits
                                    )
                                    records.append(record)
                                    ap_rows.append(values)
        print(f"graph_grid_done candidates={len(records)}", flush=True)

    ap_matrix = np.vstack(ap_rows)
    fold_of_row = np.empty(len(calibration), dtype=np.int32)
    selected = []
    nested = np.empty(len(calibration), dtype=np.float64)
    for fold, (_, validation) in enumerate(splits):
        tune = np.flatnonzero(fold_of_row != fold) if fold else None
        # Explicit construction avoids reading uninitialised fold_of_row.
        tune = np.concatenate([other_validation for index, (_, other_validation) in enumerate(splits) if index != fold])
        tune_mean = ap_matrix[:, tune].mean(axis=1)
        # Robust tie-break: prefer the configuration with the best worst-fold
        # delta on the four tuning folds, then fewer changed rows.
        tune_folds = [index for index in range(5) if index != fold]
        tune_worst = np.asarray(
            [min(record["fold_delta"][index] for index in tune_folds) for record in records]
        )
        objective = tune_mean + 0.20 * tune_worst
        best = int(np.argmax(objective))
        nested[validation] = ap_matrix[best, validation]
        selected.append({"outer_fold": fold, "candidate": best, **records[best]})

    ranked_mean = sorted(range(len(records)), key=lambda index: records[index]["map"], reverse=True)
    ranked_robust = sorted(
        range(len(records)),
        key=lambda index: (records[index]["min_fold_delta"], records[index]["map"]),
        reverse=True,
    )
    payload = {
        "baseline": {
            "map": float(base_ap.mean()),
            "fold_map": base_folds,
        },
        "html_graph": {
            "directed_edges": int(graph.sum()),
            "sources": int((graph.sum(axis=1) > 0).sum()),
            "targets": int((graph.sum(axis=0) > 0).sum()),
        },
        "candidate_count": len(records),
        "nested_selection": {
            "map": float(nested.mean()),
            "delta": float(nested.mean() - base_ap.mean()),
            "fold_map": fold_values(nested, splits),
            "selected": selected,
        },
        "top_by_mean": [{"candidate": index, **records[index]} for index in ranked_mean[:30]],
        "top_by_worst_fold": [
            {"candidate": index, **records[index]} for index in ranked_robust[:30]
        ],
    }
    Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        "exp_set_search.npz",
        ap=ap_matrix,
        base_ap=base_ap,
        nested_ap=nested,
    )
    print(
        f"nested={nested.mean():.9f} delta={nested.mean()-base_ap.mean():+.9f} "
        f"report={args.report}",
        flush=True,
    )
    for index in ranked_mean[:10]:
        record = records[index]
        print(
            f"candidate={index} family={record['family']} map={record['map']:.9f} "
            f"delta={record['delta']:+.9f} min_fold={record['min_fold_delta']:+.9f} "
            f"params={record['params']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
