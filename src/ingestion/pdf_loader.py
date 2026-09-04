from pathlib import Path
import pymupdf

from src.ingestion.chunker import chunk_text


def load_pdf_pages(file_path):
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    try:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "filename": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page_index + 1,
                    "text": text,
                }
            )
    finally:
        document.close()

    return pages


def load_pdf_chunks(file_path):
    pages = load_pdf_pages(file_path)

    chunks = []

    global_chunk_id = 0

    for page in pages:
        if not page["text"]:
            continue

        page_chunks = chunk_text(page["text"])

        for page_chunk_id, text in enumerate(page_chunks):
            chunks.append(
                {
                    "filename": page["filename"],
                    "source_path": page["source_path"],
                    "page": page["page"],
                    "chunk_id": global_chunk_id,
                    "page_chunk_id": page_chunk_id,
                    "text": text,
                }
            )

            global_chunk_id += 1

    return chunks