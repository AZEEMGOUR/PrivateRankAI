import gc
import sys
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
        document_chunks = chunk_text(document["text"])

        for chunk_id, text in enumerate(document_chunks):
            chunks.append({
                "filename": document["filename"],
                "chunk_id": chunk_id,
                "text": text
            })

    return chunks


def retrieve(query, chunks, top_k=4):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {}
    )

    texts = [chunk["text"] for chunk in chunks]

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    query_embedding = model.encode(
        [query],
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    scores = (query_embedding @ document_embeddings.T)[0]

    top_k = min(top_k, len(chunks))

    indices = torch.topk(
        scores,
        k=top_k
    ).indices.tolist()

    candidates = []

    for index in indices:
        candidate = chunks[index].copy()
        candidate["retrieval_score"] = float(scores[index])

        candidates.append(candidate)

    # Free GPU memory before loading reranker
    del model
    del document_embeddings
    del query_embedding
    del scores

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return candidates


def rerank(query, candidates):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading reranker...")

    model = CrossEncoder(
        RERANKER_MODEL,
        device=device
    )

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return candidates


def main():

    query = "How many paid leaves can an employee take?"

    print("\nQUERY:")
    print(query)

    documents = load_documents("data/sample")
    chunks = build_chunks(documents)

    print("\nDocuments:", len(documents))
    print("Chunks:", len(chunks))

    candidates = retrieve(
        query,
        chunks,
        top_k=4
    )

    print("\nRETRIEVAL RESULTS:")

    for rank, item in enumerate(candidates, 1):
        print(
            f"{rank}. "
            f"{item['filename']} "
            f"| score={item['retrieval_score']:.4f}"
        )

    final_results = rerank(
        query,
        candidates
    )

    print("\nFINAL PRIVATE RANK RESULTS:")

    for rank, item in enumerate(final_results, 1):

        print("\n------------------------------")

        print("Rank:", rank)
        print("File:", item["filename"])
        print(
            "Retrieval Score:",
            round(item["retrieval_score"], 4)
        )
        print(
            "Reranker Score:",
            round(item["rerank_score"], 4)
        )
        print("Text:", item["text"])


if __name__ == "__main__":
    main()