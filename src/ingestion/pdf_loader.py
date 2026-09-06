from pathlib import Path

import pymupdf

from src.ingestion.chunker import chunk_text


def _normalize_block_text(text):
    """
    Convert wrapped PDF lines inside one layout block
    into clean readable text.
    """

    lines = [
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip()
    ]

    return " ".join(lines)


def _looks_like_heading(text):
    """
    Detect short heading-style layout blocks.

    Examples:
    Annual Leave Policy
    Current policy - permanent employees
    Old policy - superseded
    """

    words = text.split()

    if not text:
        return False

    if len(text) > 120:
        return False

    if len(words) > 12:
        return False

    if text.endswith((".", "?", "!", ";")):
        return False

    return True


def _merge_layout_blocks(blocks):
    """
    Merge heading blocks with the body block that follows them.

    Example:

    Annual Leave Policy
    Current policy - permanent employees
    Permanent employees receive 24 days...

    becomes one semantic section.
    """

    sections = []

    pending_headings = []

    for block_text in blocks:

        if _looks_like_heading(block_text):
            pending_headings.append(
                block_text
            )
            continue

        if pending_headings:
            section_text = " ".join(
                pending_headings
                + [block_text]
            )

            pending_headings = []
        else:
            section_text = block_text

        sections.append(
            section_text
        )

    if pending_headings:
        sections.append(
            " ".join(
                pending_headings
            )
        )

    return sections


def load_pdf_pages(file_path):
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    document = pymupdf.open(
        pdf_path
    )

    pages = []

    try:
        for page_index, page in enumerate(
            document
        ):
            raw_blocks = page.get_text(
                "blocks",
                sort=True,
            )

            text_blocks = []

            for block in raw_blocks:

                # PyMuPDF block structure:
                # x0, y0, x1, y1, text,
                # block_number, block_type
                block_text = block[4]

                block_type = (
                    block[6]
                    if len(block) > 6
                    else 0
                )

                # 0 = text block
                if block_type != 0:
                    continue

                cleaned_text = (
                    _normalize_block_text(
                        block_text
                    )
                )

                if cleaned_text:
                    text_blocks.append(
                        cleaned_text
                    )

            pages.append(
                {
                    "filename": pdf_path.name,
                    "source_path": str(
                        pdf_path
                    ),
                    "page": page_index + 1,
                    "blocks": text_blocks,
                    "text": "\n".join(
                        text_blocks
                    ),
                }
            )

    finally:
        document.close()

    return pages


def load_pdf_chunks(file_path):
    pages = load_pdf_pages(
        file_path
    )

    chunks = []

    global_chunk_id = 0

    for page in pages:

        if not page["blocks"]:
            continue

        semantic_sections = (
            _merge_layout_blocks(
                page["blocks"]
            )
        )

        page_chunk_id = 0

        for section_index, section in enumerate(
            semantic_sections
        ):

            section_chunks = chunk_text(
                section,
                chunk_size=160,
                overlap=20,
            )

            for text in section_chunks:

                chunks.append(
                    {
                        "filename": (
                            page["filename"]
                        ),
                        "source_path": (
                            page[
                                "source_path"
                            ]
                        ),
                        "page": (
                            page["page"]
                        ),
                        "chunk_id": (
                            global_chunk_id
                        ),
                        "page_chunk_id": (
                            page_chunk_id
                        ),
                        "section_id": (
                            section_index
                        ),
                        "text": text,
                    }
                )

                global_chunk_id += 1
                page_chunk_id += 1

    return chunks