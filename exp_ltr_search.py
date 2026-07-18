from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRanker

from bge_pair_pilot import ap10, truths
from ltr_solution import (
    make_examples,
    query_matrices,
    query_similarity,
    label_query_scores,
    retrieval_features,
)
from solution import normalize_text


SEED = 2466955


def metric(scores, article_ids, labels, rows):
    order = np.argsort(-scores[rows], axis=1, kind="stable")[:, :10]
    return float(np.mean([
        ap10(article_ids[pred], set(labels[int(row)]))
        for pred, row in zip(order, rows)
    ]))


def main():
    data = Path("candidate_data")
    articles = pd.read_feather(data / "articles.f")
    calibration = pd.read_feather(data / "calibration.f")
    test = pd.read_feather(data / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = truths(calibration.ground_truth)
    y = np.zeros((len(calibration), len(article_ids)), dtype=np.int8)
    for row, values in enumerate(labels):
        for value in values:
            y[row, article_to_col[value]] = 1
    cal_queries = [normalize_text(value) for value in calibration.query_text]
    test_queries = [normalize_text(value) for value in test.query_text]
    components, _ = retrieval_features(articles, cal_queries, test_queries)
    word, char = query_matrices(calibration, test)
    article_static = np.column_stack((
        np.log1p(articles.title.astype(str).str.len().to_numpy()),
        np.log1p(articles.body.astype(str).str.len().to_numpy()),
    )).astype(np.float32)
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(500)))
    prepared = []
    for fold, (train, validation) in enumerate(folds):
        train_sim = query_similarity(word, char, train, train)
        np.fill_diagonal(train_sim, -1.0)
        validation_sim = query_similarity(word, char, validation, train)
        train_query = label_query_scores(train_sim, train, y)
        validation_query = label_query_scores(validation_sim, train, y)
        frequency = y[train].sum(axis=0).astype(np.float32)
        x_train, y_train, qid, _ = make_examples(
            train, components, train_query, frequency, article_static, y
        )
        x_val, _, _, candidates = make_examples(
            validation, components, validation_query, frequency, article_static
        )
        prepared.append((train, validation, x_train, y_train, qid, x_val, candidates))
        print(f"prepared fold={fold} train_examples={len(y_train)} val_examples={len(x_val)}", flush=True)

    configs = []
    for objective, depth, child, estimators, rate in itertools.product(
        ("rank:ndcg", "rank:map"),
        (3, 4, 5, 6),
        (1.0, 2.0, 4.0),
        (300, 550),
        (0.03, 0.055),
    ):
        # A compact deterministic subset covering capacity/regularization.
        if (estimators == 300) != (rate == 0.055):
            continue
        if depth == 6 and child == 1.0:
            continue
        configs.append((objective, depth, child, estimators, rate))

    results = []
    for ci, (objective, depth, child, estimators, rate) in enumerate(configs):
        oof = np.full((500, len(article_ids)), -100.0, dtype=np.float32)
        fold_values = []
        for fold, (_, validation, x_train, y_train, qid, x_val, candidates) in enumerate(prepared):
            model = XGBRanker(
                objective=objective,
                eval_metric="map@10",
                n_estimators=estimators,
                learning_rate=rate,
                max_depth=depth,
                min_child_weight=child,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                reg_alpha=0.05,
                tree_method="hist",
                n_jobs=8,
                lambdarank_pair_method="topk",
                lambdarank_num_pair_per_sample=16,
                random_state=SEED + fold,
            )
            model.fit(x_train, y_train, qid=qid, verbose=False)
            pred = model.predict(x_val)
            offset = 0
            for row, cols in zip(validation, candidates):
                oof[int(row), cols] = pred[offset: offset + len(cols)]
                offset += len(cols)
            fold_values.append(metric(oof, article_ids, labels, validation))
        value = metric(oof, article_ids, labels, np.arange(500))
        result = (value, float(np.std(fold_values)), objective, depth, child, estimators, rate, fold_values)
        results.append(result)
        print(f"config={ci + 1}/{len(configs)} {result}", flush=True)
        best = max(results, key=lambda row: row[0] - 0.20 * row[1])
        np.savez_compressed(
            "cache/ltr/exp_search_best.npz",
            scores=oof if result == best else np.load("cache/ltr/exp_search_best.npz")["scores"],
            summary=np.asarray(str(best)),
        )
    print("TOP")
    for row in sorted(results, reverse=True)[:15]:
        print(row)


if __name__ == "__main__":
    main()
