import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdf_chunks


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = PROJECT_ROOT / "models" / "private_rank_v3"
PDF_FOLDER = PROJECT_ROOT / "data" / "real_documents"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5


def load_all_pdf_chunks():
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PDF_FOLDER}"
        )

    all_chunks = []

    for pdf_file in pdf_files:
        chunks = load_pdf_chunks(pdf_file)
        all_chunks.extend(chunks)

        print(
            f"Loaded {pdf_file.name}: "
            f"{len(chunks)} chunks"
        )

    return all_chunks


def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrieve_candidates(query, chunks, device):
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {},
    )

    document_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating document embeddings...")

    document_embeddings = embedding_model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

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

    top_k = min(
        RETRIEVAL_TOP_K,
        len(chunks),
    )

    top_indices = torch.topk(
        scores,
        k=top_k,
    ).indices.tolist()

    candidates = []

    for index in top_indices:
        candidate = dict(chunks[index])

        candidate["retrieval_score"] = float(
            scores[index].item()
        )

        candidates.append(candidate)

    del query_embedding
    del document_embeddings
    del embedding_model

    clear_gpu()

    return candidates


def rerank_candidates(query, candidates, device):
    if not RERANKER_MODEL.exists():
        raise FileNotFoundError(
            f"PrivateRank V3 model not found: "
            f"{RERANKER_MODEL}"
        )

    print("\nLoading PrivateRank V3 reranker...")

    reranker = CrossEncoder(
        str(RERANKER_MODEL),
        device=device,
    )

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        result = dict(candidate)
        result["reranker_score"] = float(score)

        ranked.append(result)

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    del reranker
    clear_gpu()

    return ranked


def print_results(query, results, elapsed_ms):
    print("\n")
    print("=" * 70)
    print("PRIVATE RANK REAL DOCUMENT SEARCH")
    print("=" * 70)

    print("Query:", query)
    print(
        "Search latency:",
        round(elapsed_ms, 2),
        "ms",
    )

    for rank, result in enumerate(
        results[:FINAL_TOP_K],
        1,
    ):
        print("\n" + "-" * 70)

        print(f"Rank: {rank}")
        print(
            "File:",
            result["filename"],
        )
        print(
            "Page:",
            result["page"],
        )
        print(
            "Chunk ID:",
            result["chunk_id"],
        )
        print(
            "Retrieval Score:",
            round(
                result["retrieval_score"],
                4,
            ),
        )
        print(
            "Reranker Score:",
            round(
                result["reranker_score"],
                4,
            ),
        )

        print("\nPassage:")
        print(result["text"])

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="Question to search in enterprise PDFs",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    chunks = load_all_pdf_chunks()

    print(
        "\nTotal searchable chunks:",
        len(chunks),
    )

    start_time = time.perf_counter()

    candidates = retrieve_candidates(
        args.query,
        chunks,
        device,
    )

    ranked_results = rerank_candidates(
        args.query,
        candidates,
        device,
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    print_results(
        args.query,
        ranked_results,
        elapsed_ms,
    )


if __name__ == "__main__":
    main()