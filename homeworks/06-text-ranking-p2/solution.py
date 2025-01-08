#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
import csv
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from numba import njit
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


def get_submission_data_and_relevant_doc_ids(data_dir: str) -> tuple[pd.DataFrame, set[str]]:
    submission_data = pd.read_csv(
        f"{data_dir}/sample_submission.csv",
        header="infer",
    )

    queries_data = pd.read_csv(
        f"{data_dir}/vkmarco-doceval-queries.tsv",
        sep='\t',
        header=None,
        names=["QueryId", "QueryText"],
    )

    submission_data = submission_data.merge(queries_data, how="left", on="QueryId")
    relevant_document_ids = set(submission_data["DocumentId"].unique())

    return submission_data, relevant_document_ids


def get_relevant_documents(data_dir: str, relevant_document_ids: set[str]) -> pd.DataFrame:
    relevant_documents_list = []

    with open(f"{data_dir}/vkmarco-docs.tsv", 'r', encoding='utf-8') as doc_file:
        document_reader = csv.reader(doc_file, delimiter='\t')
        for row in tqdm(document_reader):
            if row[0] in relevant_document_ids:
                full_text = f"{row[2]} {row[3]}"
                relevant_documents_list.append({"DocumentId": row[0], "DocumentText": full_text})

    return pd.DataFrame(relevant_documents_list)


@njit
def get_query_docs_bm25_scores(
        doc_ids: list,
        tfidf_matrix: np.ndarray,
        idf: np.ndarray,
        query_vector: np.ndarray,
        doc_lengths: np.ndarray,
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
) -> list[tuple[str, float]]:
    scores = []
    for i, doc_id in enumerate(doc_ids):
        doc_vector = tfidf_matrix[i]

        score = 0
        for term_idx, term_idf in enumerate(idf):
            term_frequency = doc_vector[term_idx]
            query_term_frequency = query_vector[term_idx]

            if query_term_frequency > 0:
                numerator = term_frequency * (k1 + 1)
                denominator = term_frequency + k1 * (1 - b + b * (doc_lengths[i] / avgdl))
                score += term_idf * (numerator / denominator)

        scores.append((doc_id, score))

    return scores


def rank_documents(need_to_rank_df) -> pd.DataFrame:
    grouped = need_to_rank_df.groupby("QueryId")

    ranked_df = []

    for query_id, group in tqdm(grouped):
        query_text = group.iloc[0]["QueryText"]
        documents = group["DocumentText"].tolist()
        doc_ids = group["DocumentId"].tolist()

        vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        tfidf_matrix = vectorizer.fit_transform(documents).toarray()
        idf = vectorizer.idf_

        doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
        avgdl = np.mean(doc_lengths)

        query_vector = vectorizer.transform([query_text]).toarray().flatten()

        scores = get_query_docs_bm25_scores(
            doc_ids,
            tfidf_matrix,
            idf,
            query_vector,
            doc_lengths,
            avgdl,
        )
        scores.sort(key=lambda x: x[1], reverse=True)

        for doc_id, _ in scores:
            ranked_df.append({
                "QueryId": query_id,
                "DocumentId": doc_id,
            })

    return pd.DataFrame(ranked_df)


def main():
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start_time = timer()

    submission_data, relevant_document_ids = get_submission_data_and_relevant_doc_ids(args.data_dir)

    relevant_documents = get_relevant_documents(args.data_dir, relevant_document_ids)
    need_to_rank_df = submission_data.merge(relevant_documents, on="DocumentId", how="left")

    result_df = rank_documents(need_to_rank_df)
    result_df.to_csv(args.submission_file, index=False)

    elapsed_time = timer() - start_time
    print(f"Finished processing in {elapsed_time:.3f} seconds")


if __name__ == "__main__":
    main()
