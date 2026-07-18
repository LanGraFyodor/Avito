"""Reproduce the final Avito RAG article ranking.

The solution is a transductive hybrid:
1. a Jina v5 text-matching graph over calibration and test queries;
2. BM25/TF-IDF retrieval over cleaned articles, structural pseudo-documents and
   overlapping article chunks;
3. a small internal-link expansion and a fold-validated label-support prior.

No query is sent to an external service.  If the model is absent, only the
open-source merged model snapshot is downloaded; repository Python code is
neither downloaded nor executed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
from pathlib import Path

# Avoid pathological OpenMP initialization on some Windows/Python builds.  The
# encoder explicitly switches PyTorch to up to eight inference threads later.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from lxml import html as lxml_html
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


MODEL_ID = "jinaai/jina-embeddings-v5-text-small-text-matching"
MODEL_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.jinja",
    "1_Pooling/*",
)
ARTICLE_LINK_RE = re.compile(r"(?:/articles/|articleId=)([0-9]+)")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^0-9a-zа-я<>]+", re.IGNORECASE)
UNSEEN_FACTOR = 0.55


def normalize_text(value: object) -> str:
    text = html.unescape(str(value)).lower().replace("ё", "е")
    text = text.replace("<money>", " деньги ").replace("<date>", " дата ")
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def parse_article(title: object, body: object) -> tuple[str, str]:
    raw_body = str(body)
    try:
        root = lxml_html.fromstring(raw_body)
        for tag in root.xpath("//script|//style|//noscript|//svg"):
            tag.drop_tree()
        structural = " ".join(
            " ".join(tag.itertext())
            for tag in root.xpath(
                "//h1|//h2|//h3|//h4|//h5|//strong|//label|//summary|//th"
            )
        )
        visible = " ".join(root.itertext())
    except (ValueError, lxml_html.ParserError):
        structural = ""
        visible = re.sub(r"<[^>]+>", " ", raw_body)
    title_text = normalize_text(title)
    focus = normalize_text(f"{title_text} {title_text} {structural}")
    full = normalize_text(
        f"{title_text} {title_text} {title_text} {structural} {visible}"
    )
    return full, focus


def ground_truth_lists(values) -> list[list[int]]:
    return [[int(item) for item in str(value).split()] for value in values]


def row_max_scale(scores: np.ndarray) -> np.ndarray:
    scores = np.maximum(scores.astype(np.float32, copy=False), 0.0)
    maximum = scores.max(axis=1, keepdims=True)
    return np.divide(scores, maximum, out=np.zeros_like(scores), where=maximum > 0)


def tfidf_pair(
    documents: list[str],
    calibration_queries: list[str],
    test_queries: list[str],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    texts = documents + calibration_queries + test_queries
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w-]{2,}\b" if analyzer == "word" else None,
        dtype=np.float32,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    n_documents = len(documents)
    n_calibration = len(calibration_queries)
    document_matrix = matrix[:n_documents]
    query_matrix = matrix[n_documents:]
    scores = (query_matrix @ document_matrix.T).toarray().astype(np.float32)
    return scores[:n_calibration], scores[n_calibration:]


def bm25_pair(
    documents: list[str],
    calibration_queries: list[str],
    test_queries: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.995,
        max_features=240_000,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        dtype=np.float32,
    )
    document_counts = vectorizer.fit_transform(documents).tocsr()
    query_counts = vectorizer.transform(calibration_queries + test_queries).tocsr()
    query_counts.data[:] = 1.0
    n_documents = document_counts.shape[0]
    document_frequency = np.asarray((document_counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(
        (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)
    lengths = np.asarray(document_counts.sum(axis=1)).ravel()
    average_length = max(float(lengths.mean()), 1.0)
    k1, b = 1.5, 0.75
    row_ids = np.repeat(np.arange(n_documents), np.diff(document_counts.indptr))
    length_norm = k1 * (1.0 - b + b * lengths[row_ids] / average_length)
    document_counts.data = (
        inverse_document_frequency[document_counts.indices]
        * document_counts.data
        * (k1 + 1.0)
        / (document_counts.data + length_norm)
    )
    scores = (query_counts @ document_counts.T).toarray().astype(np.float32)
    n_calibration = len(calibration_queries)
    return scores[:n_calibration], scores[n_calibration:]


def extract_article_corpus(
    articles: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], np.ndarray, sparse.csr_matrix]:
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    article_to_col = {int(value): index for index, value in enumerate(article_ids)}
    titles = [normalize_text(value) for value in articles.title]
    full_texts: list[str] = []
    structural: list[str] = []
    chunks: list[str] = []
    chunk_cols: list[int] = []
    inbound: list[list[str]] = [[] for _ in range(len(articles))]
    edge_rows: list[int] = []
    edge_cols: list[int] = []

    for article_col, row in enumerate(articles.itertuples()):
        full, focus = parse_article(row.title, row.body)
        full_texts.append(full)
        try:
            root = lxml_html.fromstring(str(row.body))
            heading_parts = [
                " ".join(element.itertext())
                for element in root.xpath(
                    "//h1|//h2|//h3|//h4|//summary|//label|//th|//*[@data-tab-name]"
                )
            ]
            tab_names = [
                element.get("data-tab-name", "")
                for element in root.xpath("//*[@data-tab-name]")
            ]
            visible = normalize_text(" ".join(root.itertext()))
            for anchor in root.xpath("//a[@href]"):
                match = ARTICLE_LINK_RE.search(anchor.get("href", ""))
                if not match:
                    continue
                target_id = int(match.group(1))
                if target_id not in article_to_col or target_id == int(row.article_id):
                    continue
                target_col = article_to_col[target_id]
                anchor_text = normalize_text(" ".join(anchor.itertext()))
                inbound[target_col].append(
                    normalize_text(f"{titles[article_col]} {anchor_text}")
                )
                edge_rows.append(article_col)
                edge_cols.append(target_col)
        except (ValueError, lxml_html.ParserError):
            heading_parts, tab_names, visible = [], [], full
        structure = normalize_text(
            f"{titles[article_col]} {focus} {' '.join(heading_parts)} {' '.join(tab_names)}"
        )
        structural.append(structure)
        words = visible.split()
        for start in (list(range(0, max(len(words), 1), 70))[:18] or [0]):
            chunk = " ".join(words[start : start + 100])
            chunks.append(normalize_text(f"{titles[article_col]} {structure} {chunk}"))
            chunk_cols.append(article_col)

    pseudo_documents = [
        normalize_text(
            f"{title} {title} {title} {structure} "
            + " ".join(text for text in inbound[index] for _ in range(2))
        )
        for index, (title, structure) in enumerate(zip(titles, structural))
    ]
    data = np.ones(len(edge_rows) * 2, dtype=np.float32)
    rows = np.asarray(edge_rows + edge_cols, dtype=np.int64)
    cols = np.asarray(edge_cols + edge_rows, dtype=np.int64)
    graph = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(articles), len(articles)), dtype=np.float32
    )
    graph.data[:] = 1.0
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.divide(
        1.0,
        degree,
        out=np.zeros_like(degree, dtype=np.float32),
        where=degree > 0,
    )
    return full_texts, pseudo_documents, chunks, np.asarray(chunk_cols), sparse.diags(inverse) @ graph


def chunk_scores(
    chunks: list[str],
    chunk_cols: np.ndarray,
    calibration_queries: list[str],
    test_queries: list[str],
    n_articles: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.995,
        max_features=220_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        dtype=np.float32,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(chunks + calibration_queries + test_queries)
    similarities = (matrix[len(chunks) :] @ matrix[: len(chunks)].T).toarray().astype(np.float32)
    scores = np.zeros((len(calibration_queries) + len(test_queries), n_articles), dtype=np.float32)
    for article_col in range(n_articles):
        indices = np.flatnonzero(chunk_cols == article_col)
        scores[:, article_col] = similarities[:, indices].max(axis=1)
    return scores[: len(calibration_queries)], scores[len(calibration_queries) :]


def article_scores(
    articles: pd.DataFrame,
    calibration_queries: list[str],
    test_queries: list[str],
) -> np.ndarray:
    full, pseudo, chunks, chunk_cols, graph = extract_article_corpus(articles)
    pairs = [
        bm25_pair(full, calibration_queries, test_queries),
        tfidf_pair(
            pseudo,
            calibration_queries,
            test_queries,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=220_000,
        ),
        tfidf_pair(
            pseudo,
            calibration_queries,
            test_queries,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=220_000,
        ),
        chunk_scores(
            chunks,
            chunk_cols,
            calibration_queries,
            test_queries,
            len(articles),
        ),
    ]
    weights = (1.0, 0.5, 0.5, 0.5)
    test_base = sum(weight * row_max_scale(pair[1]) for weight, pair in zip(weights, pairs))
    one_hop = row_max_scale(np.asarray(test_base @ graph))
    two_hop = row_max_scale(np.asarray(one_hop @ graph))
    return test_base + 0.50 * one_hop + 0.25 * two_hop


def ensure_model(model_dir: Path) -> None:
    if (model_dir / "config.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=model_dir,
        allow_patterns=list(MODEL_PATTERNS),
    )


def query_embeddings(
    texts: list[str], model_dir: Path, cache_path: Path
) -> np.ndarray:
    if cache_path.exists():
        payload = np.load(cache_path)
        cached = (
            payload["embeddings"]
            if "embeddings" in payload.files
            else np.vstack([payload["cal"], payload["test"]])
        )
        if len(cached) == len(texts):
            return cached
    ensure_model(model_dir)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = 128
    embeddings = model.encode(
        texts,
        batch_size=8,
        prompt_name="document",
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def query_graph_scores(
    embeddings: np.ndarray,
    truths: list[list[int]],
    article_ids: np.ndarray,
    n_calibration: int,
) -> np.ndarray:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, 0.0)
    # Rank fusion is more robust than raw cosine across heterogeneous query styles.
    rank_similarity = np.zeros_like(similarity, dtype=np.float32)
    candidate_count = min(160, len(similarity) - 1)
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :candidate_count]
    values = 1.0 / np.sqrt(np.arange(1, candidate_count + 1, dtype=np.float32))
    rank_similarity[np.arange(len(similarity))[:, None], order] = values[None, :]
    np.fill_diagonal(rank_similarity, 0.0)

    neighbors = 15
    columns = np.argpartition(rank_similarity, -neighbors, axis=1)[:, -neighbors:]
    graph_values = np.take_along_axis(rank_similarity, columns, axis=1)
    rows = np.repeat(np.arange(len(similarity), dtype=np.int64), neighbors)
    graph = sparse.csr_matrix(
        (graph_values.reshape(-1), (rows, columns.reshape(-1))),
        shape=similarity.shape,
        dtype=np.float32,
    )
    graph = graph.maximum(graph.T).tocsr()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.divide(
        1.0,
        degree,
        out=np.zeros_like(degree, dtype=np.float32),
        where=degree > 0,
    )
    transition = sparse.diags(inverse) @ graph

    article_to_col = {int(value): index for index, value in enumerate(article_ids)}
    initial = np.zeros((len(embeddings), len(article_ids)), dtype=np.float32)
    for row, labels in enumerate(truths):
        for article_id in labels:
            initial[row, article_to_col[article_id]] = 1.0
    train_rows = np.arange(n_calibration, dtype=np.int64)
    scores = initial.copy()
    alpha = 0.90
    for _ in range(60):
        updated = alpha * (transition @ scores) + (1.0 - alpha) * initial
        updated[train_rows] = initial[train_rows]
        if float(np.max(np.abs(updated - scores))) < 1.0e-6:
            scores = updated
            break
        scores = updated
    return row_max_scale(scores[n_calibration:])


def validate_answer(path: Path, test: pd.DataFrame, article_ids: np.ndarray) -> str:
    answer = pd.read_csv(path, dtype={"answer": "string"})
    if answer.columns.tolist() != ["query_id", "answer"]:
        raise AssertionError("Expected exactly query_id and answer columns")
    if answer.query_id.tolist() != test.query_id.tolist():
        raise AssertionError("query_id values/order differ from test.f")
    valid = set(map(int, article_ids))
    for value in answer.answer:
        ids = list(map(int, str(value).split()))
        if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= valid:
            raise AssertionError("Invalid article list in answer.csv")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("candidate_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/jina-v5-text-matching-merged"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    args = parser.parse_args()

    articles = pd.read_feather(args.data_dir / "articles.f")
    calibration = pd.read_feather(args.data_dir / "calibration.f")
    test = pd.read_feather(args.data_dir / "test.f")
    article_ids = articles.article_id.to_numpy(dtype=np.int64)
    truths = ground_truth_lists(calibration.ground_truth)
    calibration_queries = [normalize_text(value) for value in calibration.query_text]
    test_queries = [normalize_text(value) for value in test.query_text]
    all_queries = calibration_queries + test_queries

    embeddings = query_embeddings(
        all_queries,
        args.model_dir,
        args.cache_dir / "jina_v5_text_matching_query_embeddings.npz",
    )
    query_scores = query_graph_scores(
        embeddings, truths, article_ids, len(calibration)
    )
    document_scores = row_max_scale(
        article_scores(articles, calibration_queries, test_queries)
    )
    scores = query_scores + document_scores

    seen_ids = {value for labels in truths for value in labels}
    seen_mask = np.asarray([int(value) in seen_ids for value in article_ids])
    scores[:, ~seen_mask] *= UNSEEN_FACTOR
    columns = np.argpartition(scores, -10, axis=1)[:, -10:]
    values = np.take_along_axis(scores, columns, axis=1)
    columns = np.take_along_axis(
        columns, np.argsort(-values, axis=1, kind="stable"), axis=1
    )
    predictions = article_ids[columns]
    answer = test[["query_id"]].copy()
    answer["answer"] = [" ".join(map(str, row)) for row in predictions]
    answer.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    digest = validate_answer(args.output, test, article_ids)
    print(f"saved {args.output.resolve()} rows={len(answer)} sha256={digest}")


if __name__ == "__main__":
    main()
