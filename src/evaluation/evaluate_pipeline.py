import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.document_loader import load_documents
from src.ingestion.chunker import chunk_text


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def build_chunks(documents):
    chunks = []

    for document in documents:
        for chunk_id, text in enumerate(chunk_text(document["text"])):
            chunks.append(
                {
                    "filename": document["filename"],
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

    return chunks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------
    # Load documents
    # -----------------------------

    documents = load_documents("data/sample")
    chunks = build_chunks(documents)

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))

    # -----------------------------
    # Load evaluation queries
    # -----------------------------

    with open(
        "data/evaluation/queries.json",
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    print("Evaluation Queries:", len(queries))

    # -----------------------------
    # Load models once
    # -----------------------------

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {},
    )

    print("Creating document embeddings...")

    document_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    document_embeddings = embedding_model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    print("Loading reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=device,
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    retrieval_top1_correct = 0
    retrieval_top3_correct = 0
    reranker_top1_correct = 0

    reciprocal_ranks = []
    latencies = []

    print("\n==============================")
    print("EVALUATION STARTED")
    print("==============================")

    for number, item in enumerate(queries, 1):
        query = item["query"]
        expected_file = item["expected_file"]

        start_time = time.perf_counter()

        # -------------------------
        # Retrieval
        # -------------------------

        query_embedding = embedding_model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        scores = (
            query_embedding
            @ document_embeddings.T
        )[0]

        ranked_indices = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        retrieval_files = [
            chunks[index]["filename"]
            for index in ranked_indices
        ]

        if retrieval_files[0] == expected_file:
            retrieval_top1_correct += 1

        if expected_file in retrieval_files[:3]:
            retrieval_top3_correct += 1

        # -------------------------
        # Reranking
        # -------------------------

        top_k = min(10, len(ranked_indices))

        candidate_indices = ranked_indices[:top_k]

        candidates = [
            chunks[index]
            for index in candidate_indices
        ]

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        reranker_scores = reranker.predict(pairs)

        ranked_candidates = sorted(
            zip(candidates, reranker_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        final_files = [
            candidate["filename"]
            for candidate, _ in ranked_candidates
        ]

        if final_files[0] == expected_file:
            reranker_top1_correct += 1

        if expected_file in final_files:
            expected_rank = (
                final_files.index(expected_file) + 1
            )

            reciprocal_ranks.append(
                1 / expected_rank
            )
        else:
            reciprocal_ranks.append(0)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        latencies.append(elapsed_ms)

        print(f"\nQuery {number}: {query}")
        print("Expected:", expected_file)
        print("Retrieval #1:", retrieval_files[0])
        print("Reranker #1:", final_files[0])
        print(
            "Latency:",
            round(elapsed_ms, 2),
            "ms",
        )

    # -----------------------------
    # Final metrics
    # -----------------------------

    total = len(queries)

    retrieval_recall_1 = (
        retrieval_top1_correct / total
    )

    retrieval_recall_3 = (
        retrieval_top3_correct / total
    )

    reranker_recall_1 = (
        reranker_top1_correct / total
    )

    mrr = sum(reciprocal_ranks) / total

    avg_latency = sum(latencies) / total

    print("\n==============================")
    print("PRIVATE RANK EVALUATION")
    print("==============================")

    print("Total Queries:", total)

    print(
        "Retrieval Recall@1:",
        f"{retrieval_recall_1:.2%}",
    )

    print(
        "Retrieval Recall@3:",
        f"{retrieval_recall_3:.2%}",
    )

    print(
        "Reranker Recall@1:",
        f"{reranker_recall_1:.2%}",
    )

    print(
        "MRR:",
        round(mrr, 4),
    )

    print(
        "Average Query Latency:",
        round(avg_latency, 2),
        "ms",
    )


if __name__ == "__main__":
    main()