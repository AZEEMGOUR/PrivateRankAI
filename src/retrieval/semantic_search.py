import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.document_loader import load_documents
from src.ingestion.chunker import chunk_text


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def build_chunks(documents):
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": document["filename"],
                "chunk_id": index,
                "text": chunk,
            })

    return all_chunks


def main():
    print("CUDA available:", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {}
    )

    documents = load_documents("data/sample")
    chunks = build_chunks(documents)

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))

    texts = [chunk["text"] for chunk in chunks]

    print("\nCreating document embeddings...")

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    query = "How many paid leaves are employees allowed?"

    print("\nQuery:", query)

    query_embedding = model.encode(
        [query],
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    scores = query_embedding @ document_embeddings.T

    scores = scores[0]

    ranked_indices = torch.argsort(
        scores,
        descending=True
    )

    print("\nSEMANTIC SEARCH RESULTS:")

    for rank, index in enumerate(ranked_indices, 1):
        index = index.item()

        result = chunks[index]

        print("\n------------------------------")
        print("Rank:", rank)
        print("Score:", round(scores[index].item(), 4))
        print("File:", result["filename"])
        print("Text:", result["text"])


if __name__ == "__main__":
    main()