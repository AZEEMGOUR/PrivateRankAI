import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


INDEX_FOLDER = PROJECT_ROOT / "data" / "indexes"
CHUNKS_FILE = INDEX_FOLDER / "chunks.json"
EMBEDDINGS_FILE = INDEX_FOLDER / "embeddings.pt"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5


def load_index(device):
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {CHUNKS_FILE}\n"
            "Run scripts/build_document_index.py first."
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding index not found: {EMBEDDINGS_FILE}\n"
            "Run scripts/build_document_index.py first."
        )

    print("Loading saved document index...")

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    embeddings = torch.load(
        EMBEDDINGS_FILE,
        map_location=device,
        weights_only=True,
    )

    print("Indexed chunks:", len(chunks))
    print(
        "Embedding shape:",
        tuple(embeddings.shape),
    )

    return chunks, embeddings


def load_models(device, reranker_model):
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {},
    )

    reranker_path = PROJECT_ROOT / reranker_model

    if reranker_path.exists():
        reranker_source = str(reranker_path)
    else:
        reranker_source = reranker_model

    print("Loading reranker:", reranker_source)

    reranker = CrossEncoder(
        reranker_source,
        device=device,
    )

    return embedding_model, reranker


def retrieve_candidates(
    query,
    chunks,
    document_embeddings,
    embedding_model,
):
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

    return candidates


def rerank_candidates(
    query,
    candidates,
    reranker,
):
    pairs = [
        (
            query,
            candidate["text"],
        )
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        result = dict(candidate)

        result["reranker_score"] = float(
            score
        )

        ranked.append(result)

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return ranked


def print_results(
    query,
    results,
    elapsed_ms,
):
    print("\n")
    print("=" * 70)
    print("PRIVATE RANK REAL DOCUMENT SEARCH")
    print("=" * 70)

    print("Query:", query)

    print(
        "Query latency:",
        round(elapsed_ms, 2),
        "ms",
    )

    for rank, result in enumerate(
        results[:FINAL_TOP_K],
        1,
    ):
        print("\n" + "-" * 70)

        print("Rank:", rank)
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


def search(
    query,
    chunks,
    document_embeddings,
    embedding_model,
    reranker,
):
    start_time = time.perf_counter()

    candidates = retrieve_candidates(
        query,
        chunks,
        document_embeddings,
        embedding_model,
    )

    results = rerank_candidates(
        query,
        candidates,
        reranker,
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    print_results(
        query,
        results,
        elapsed_ms,
    )


def interactive_search(
    chunks,
    document_embeddings,
    embedding_model,
    reranker,
):
    print("\n")
    print("=" * 70)
    print("PRIVATE RANK INTERACTIVE SEARCH")
    print("=" * 70)

    print(
        "Models and document index are loaded."
    )

    print(
        "Type a question and press Enter."
    )

    print(
        "Type 'exit' to close."
    )

    while True:
        query = input("\nSearch > ").strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Search closed.")
            break

        search(
            query,
            chunks,
            document_embeddings,
            embedding_model,
            reranker,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        help=(
            "Optional single search query. "
            "Without this argument interactive "
            "search starts."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default="models/private_rank_v3",
        help="Local reranker path or Hugging Face model name",
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

    startup_start = time.perf_counter()

    chunks, document_embeddings = (
        load_index(device)
    )

    embedding_model, reranker = (
        load_models(
            device,
            args.reranker_model,
        )
    )

    startup_ms = (
        time.perf_counter()
        - startup_start
    ) * 1000

    print(
        "\nStartup time:",
        round(startup_ms, 2),
        "ms",
    )

    if args.query:
        search(
            args.query,
            chunks,
            document_embeddings,
            embedding_model,
            reranker,
        )
    else:
        interactive_search(
            chunks,
            document_embeddings,
            embedding_model,
            reranker,
        )


if __name__ == "__main__":
    main()