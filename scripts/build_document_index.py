import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdf_chunks


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

DOCUMENT_FOLDER = PROJECT_ROOT / "data" / "real_documents"
INDEX_FOLDER = PROJECT_ROOT / "data" / "indexes"

CHUNKS_FILE = INDEX_FOLDER / "chunks.json"
EMBEDDINGS_FILE = INDEX_FOLDER / "embeddings.pt"


def load_all_chunks():
    pdf_files = sorted(DOCUMENT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {DOCUMENT_FOLDER}"
        )

    chunks = []

    for pdf_file in pdf_files:
        pdf_chunks = load_pdf_chunks(pdf_file)

        print(
            f"Loaded {pdf_file.name}: "
            f"{len(pdf_chunks)} chunks"
        )

        chunks.extend(pdf_chunks)

    return chunks


def main():
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

    INDEX_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_all_chunks()

    print(
        "\nTotal chunks:",
        len(chunks),
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        } if device == "cuda" else {},
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.cpu()

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    torch.save(
        embeddings,
        EMBEDDINGS_FILE,
    )

    print("\n==============================")
    print("DOCUMENT INDEX CREATED")
    print("==============================")
    print("Documents folder:", DOCUMENT_FOLDER)
    print("Chunks:", len(chunks))
    print("Embedding shape:", tuple(embeddings.shape))
    print("Metadata:", CHUNKS_FILE)
    print("Embeddings:", EMBEDDINGS_FILE)


if __name__ == "__main__":
    main()