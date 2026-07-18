from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRanker

from bge_pair_pilot import ap10, truths
from solution import (
    bm25_pair,
    chunk_scores,
    extract_article_corpus,
    normalize_text,
    row_max_scale,
    tfidf_pair,
)


DATA_DIR = Path("candidate_data")
CACHE_DIR = Path("cache/ltr")
TOP_DIRECT = 80
TOP_QUERY = 60


def evaluate(scores, article_ids, labels, rows=None):
    if rows is None:
        rows = np.arange(len(labels))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return float(
        np.mean(
            [ap10(article_ids[prediction], set(labels[int(row)])) for prediction, row in zip(order, rows)]
        )
    )


def retrieval_features(articles, calibration_queries, test_queries):
    path = CACHE_DIR / "retrieval.npz"
    if path.exists():
        payload = np.load(path)
        return [payload[f"cal_{i}"] for i in range(6)], [payload[f"test_{i}"] for i in range(6)]
    full, pseudo, chunks, chunk_cols, _ = extract_article_corpus(articles)
    titles = [normalize_text(value) for value in articles.title]
    pairs = [
        bm25_pair(full, calibration_queries, test_queries),
        tfidf_pair(
            pseudo, calibration_queries, test_queries,
            analyzer="word", ngram_range=(1, 2), max_features=220_000,
        ),
        tfidf_pair(
            pseudo, calibration_queries, test_queries,
            analyzer="char_wb", ngram_range=(3, 5), max_features=220_000,
        ),
        chunk_scores(chunks, chunk_cols, calibration_queries, test_queries, len(articles)),
        tfidf_pair(
            titles, calibration_queries, test_queries,
            analyzer="word", ngram_range=(1, 2), max_features=120_000,
        ),
        tfidf_pair(
            titles, calibration_queries, test_queries,
            analyzer="char_wb", ngram_range=(3, 5), max_features=160_000,
        ),
    ]
    cal = [row_max_scale(pair[0]) for pair in pairs]
    test = [row_max_scale(pair[1]) for pair in pairs]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    values = {f"cal_{i}": value for i, value in enumerate(cal)}
    values.update({f"test_{i}": value for i, value in enumerate(test)})
    np.savez_compressed(path, **values)
    return cal, test


def query_matrices(calibration, test):
    texts = calibration.query_text.astype(str).tolist() + test.query_text.astype(str).tolist()
    normalized = [normalize_text(value) for value in texts]
    word = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=120_000
    ).fit_transform(normalized)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=180_000
    ).fit_transform(normalized)
    return word, char


def query_similarity(word, char, targets, sources):
    return (
        0.45 * (word[targets] @ word[sources].T).toarray()
        + 0.55 * (char[targets] @ char[sources].T).toarray()
    ).astype(np.float32)


def label_query_scores(similarity, source_rows, y):
    local_y = y[source_rows]
    output = np.zeros((len(similarity), y.shape[1]), dtype=np.float32)
    for col in np.flatnonzero(local_y.sum(axis=0)):
        positives = np.flatnonzero(local_y[:, col])
        values = similarity[:, positives]
        output[:, col] = values.max(axis=1)
        if len(positives) > 1:
            top = np.sort(values, axis=1)[:, -min(3, len(positives)) :]
            output[:, col] += 0.20 * top.mean(axis=1)
    return output


def combined(components):
    weights = np.asarray([1.0, 0.55, 0.55, 0.60, 0.65, 0.65], dtype=np.float32)
    return sum(weight * values for weight, values in zip(weights, components))


def rank_percent(values):
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(values, dtype=np.float32)
    ranks[np.arange(len(values))[:, None], order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks / values.shape[1]


def make_examples(
    rows,
    components,
    query_scores,
    frequency,
    article_static,
    y=None,
):
    local_components = [values[rows] for values in components]
    direct = combined(local_components)
    direct_ranks = [rank_percent(values) for values in local_components]
    query_ranks = rank_percent(query_scores)
    features = []
    targets = []
    qids = []
    candidate_rows = []
    for local_row, global_row in enumerate(rows):
        direct_top = np.argsort(-direct[local_row], kind="stable")[:TOP_DIRECT]
        query_top = np.argsort(-query_scores[local_row], kind="stable")[:TOP_QUERY]
        candidates = np.asarray(list(dict.fromkeys(np.r_[direct_top, query_top].tolist())), dtype=np.int64)
        columns = [values[local_row, candidates] for values in local_components]
        columns += [values[local_row, candidates] for values in direct_ranks]
        columns += [
            query_scores[local_row, candidates],
            query_ranks[local_row, candidates],
            direct[local_row, candidates],
            query_scores[local_row, candidates] * direct[local_row, candidates],
            np.log1p(frequency[candidates]),
            article_static[candidates, 0],
            article_static[candidates, 1],
        ]
        features.append(np.column_stack(columns).astype(np.float32))
        candidate_rows.append(candidates)
        qids.extend([local_row] * len(candidates))
        if y is not None:
            targets.append(y[int(global_row), candidates].astype(np.float32))
    return (
        np.vstack(features),
        np.concatenate(targets) if targets else None,
        np.asarray(qids, dtype=np.int64),
        candidate_rows,
    )


def make_ranker(seed):
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="map@10",
        n_estimators=420,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        tree_method="hist",
        n_jobs=8,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=12,
        random_state=seed,
    )


def run_oof(name, splits, components, word, char, y, article_static, article_ids, labels):
    oof = np.full((len(labels), len(article_ids)), -100.0, dtype=np.float32)
    recalls = []
    for fold, (train, validation) in enumerate(splits):
        train_similarity = query_similarity(word, char, train, train)
        np.fill_diagonal(train_similarity, -1.0)
        validation_similarity = query_similarity(word, char, validation, train)
        train_query = label_query_scores(train_similarity, train, y)
        validation_query = label_query_scores(validation_similarity, train, y)
        frequency = y[train].sum(axis=0).astype(np.float32)
        x_train, y_train, qid_train, _ = make_examples(
            train, components, train_query, frequency, article_static, y
        )
        x_validation, _, _, candidates = make_examples(
            validation, components, validation_query, frequency, article_static
        )
        model = make_ranker(2466955 + fold)
        model.fit(x_train, y_train, qid=qid_train, verbose=False)
        prediction = model.predict(x_validation)
        offset = 0
        for local_row, (global_row, cols) in enumerate(zip(validation, candidates)):
            values = prediction[offset : offset + len(cols)]
            oof[int(global_row), cols] = values
            offset += len(cols)
            actual = set(np.flatnonzero(y[int(global_row)]))
            recalls.append(len(actual & set(cols)) / len(actual))
        print(
            f"{name}_fold={fold} map={evaluate(oof[validation], article_ids, labels, validation):.6f} "
            f"candidate_recall={np.mean(recalls):.6f}",
            flush=True,
        )
    value = evaluate(oof, article_ids, labels)
    print(f"{name}_map={value:.6f} candidate_recall={np.mean(recalls):.6f}")
    return oof


def fit_final(
    calibration,
    test,
    cal_components,
    test_components,
    word,
    char,
    y,
    article_static,
    article_ids,
):
    train_rows = np.arange(len(calibration))
    test_rows = np.arange(len(test))
    train_similarity = query_similarity(word, char, train_rows, train_rows)
    np.fill_diagonal(train_similarity, -1.0)
    test_global_rows = len(calibration) + test_rows
    test_similarity = query_similarity(word, char, test_global_rows, train_rows)
    train_query = label_query_scores(train_similarity, train_rows, y)
    test_query = label_query_scores(test_similarity, train_rows, y)
    frequency = y.sum(axis=0).astype(np.float32)

    x_train, y_train, qid_train, _ = make_examples(
        train_rows, cal_components, train_query, frequency, article_static, y
    )
    x_test, _, _, candidates = make_examples(
        test_rows, test_components, test_query, frequency, article_static
    )
    model = make_ranker(2466955)
    model.fit(x_train, y_train, qid=qid_train, verbose=False)
    prediction = model.predict(x_test)
    scores = np.full((len(test), len(article_ids)), -100.0, dtype=np.float32)
    offset = 0
    for row, cols in enumerate(candidates):
        scores[row, cols] = prediction[offset : offset + len(cols)]
        offset += len(cols)

    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    answers = [" ".join(map(str, article_ids[row])) for row in order]
    result = pd.DataFrame({"query_id": test.query_id, "answer": answers})
    result.to_csv("answer_ltr.csv", index=False)
    np.savez_compressed(CACHE_DIR / "test_scores.npz", scores=scores, article_ids=article_ids)
    print("saved=answer_ltr.csv rows=500 unique_query_id=500")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("standard", "group", "both", "final"), default="both")
    args = parser.parse_args()
    articles = pd.read_feather(DATA_DIR / "articles.f")
    calibration = pd.read_feather(DATA_DIR / "calibration.f")
    test = pd.read_feather(DATA_DIR / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    labels = truths(calibration.ground_truth)
    y = np.zeros((len(calibration), len(article_ids)), dtype=np.int8)
    for row, values in enumerate(labels):
        for value in values:
            y[row, article_to_col[value]] = 1
    calibration_queries = [normalize_text(value) for value in calibration.query_text]
    test_queries = [normalize_text(value) for value in test.query_text]
    cal_components, test_components = retrieval_features(
        articles, calibration_queries, test_queries
    )
    word, char = query_matrices(calibration, test)
    article_static = np.column_stack(
        (
            np.log1p(articles.title.astype(str).str.len().to_numpy()),
            np.log1p(articles.body.astype(str).str.len().to_numpy()),
        )
    ).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.only == "final":
        fit_final(
            calibration,
            test,
            cal_components,
            test_components,
            word,
            char,
            y,
            article_static,
            article_ids,
        )
        return
    if args.only in ("standard", "both"):
        standard = run_oof(
            "standard",
            list(KFold(5, shuffle=True, random_state=2466955).split(np.arange(len(calibration)))),
            cal_components,
            word,
            char,
            y,
            article_static,
            article_ids,
            labels,
        )
        np.savez_compressed(CACHE_DIR / "standard_oof.npz", scores=standard)
    if args.only in ("group", "both"):
        groups = np.asarray([" ".join(map(str, sorted(row))) for row in labels])
        group = run_oof(
            "group",
            list(GroupKFold(5).split(np.arange(len(calibration)), groups=groups)),
            cal_components,
            word,
            char,
            y,
            article_static,
            article_ids,
            labels,
        )
        np.savez_compressed(CACHE_DIR / "group_oof.npz", scores=group)


if __name__ == "__main__":
    main()
