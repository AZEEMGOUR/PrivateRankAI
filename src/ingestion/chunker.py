import re


def _clean_line(line):
    return " ".join(
        line.strip().split()
    )


def _is_heading(line):
    """
    Detect short heading-like PDF lines.

    Examples:
    2. Annual Leave Policy
    Current policy - permanent employees
    Probation employees
    Old policy - superseded
    """

    words = line.split()

    if not line:
        return False

    if len(line) > 100:
        return False

    if len(words) > 8:
        return False

    # Numbered section heading:
    # 2. Annual Leave Policy
    # 3.1 Security Policy
    if re.match(
        r"^\d+(?:\.\d+)*[.)]?\s+",
        line,
    ):
        return True

    # Normal sentence endings usually mean
    # this is body text, not a heading.
    if line.endswith(
        (
            ".",
            "?",
            "!",
            ",",
            ";",
        )
    ):
        return False

    return True


def _build_semantic_blocks(text):
    """
    Preserve PDF line structure and create
    blocks around headings instead of blindly
    slicing every N words.
    """

    lines = [
        _clean_line(line)
        for line in text.splitlines()
        if _clean_line(line)
    ]

    if not lines:
        return []

    blocks = []

    current_lines = []
    current_has_body = False

    for line in lines:

        if _is_heading(line):

            # A new heading begins after body text:
            # finish previous semantic section.
            if (
                current_lines
                and current_has_body
            ):
                blocks.append(
                    " ".join(current_lines)
                )

                current_lines = [line]
                current_has_body = False

            else:
                # Consecutive headings belong together.
                #
                # Example:
                # 2. Annual Leave Policy
                # Current policy - permanent employees
                current_lines.append(line)

        else:
            current_lines.append(line)
            current_has_body = True

    if current_lines:
        blocks.append(
            " ".join(current_lines)
        )

    return blocks


def _split_large_block(
    text,
    chunk_size,
    overlap,
):
    """
    Only split when one semantic section itself
    is too large.

    Prefer sentence boundaries. Word overlap is
    used only as a fallback for oversized text.
    """

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = sentence.split()

        # One extremely long sentence.
        if len(sentence_words) > chunk_size:

            if current_sentences:
                chunks.append(
                    " ".join(
                        current_sentences
                    )
                )

                current_sentences = []
                current_word_count = 0

            start = 0

            while start < len(
                sentence_words
            ):
                end = (
                    start
                    + chunk_size
                )

                chunks.append(
                    " ".join(
                        sentence_words[
                            start:end
                        ]
                    )
                )

                if end >= len(
                    sentence_words
                ):
                    break

                start += max(
                    1,
                    chunk_size - overlap,
                )

            continue

        if (
            current_sentences
            and current_word_count
            + len(sentence_words)
            > chunk_size
        ):
            chunks.append(
                " ".join(
                    current_sentences
                )
            )

            current_sentences = []
            current_word_count = 0

        current_sentences.append(
            sentence
        )

        current_word_count += len(
            sentence_words
        )

    if current_sentences:
        chunks.append(
            " ".join(
                current_sentences
            )
        )

    return chunks


def chunk_text(
    text,
    chunk_size=120,
    overlap=20,
):
    """
    Section-aware enterprise document chunker.

    1. Preserve headings and PDF line structure.
    2. Keep each logical section separate.
    3. Split only sections that are too large.
    4. Never overlap unrelated sections.
    """

    if not text or not text.strip():
        return []

    semantic_blocks = (
        _build_semantic_blocks(
            text
        )
    )

    chunks = []

    for block in semantic_blocks:

        block_chunks = (
            _split_large_block(
                block,
                chunk_size,
                overlap,
            )
        )

        chunks.extend(
            block_chunks
        )

    return chunks


if __name__ == "__main__":

    sample_text = """
2. Annual Leave Policy
Current policy - permanent employees
Permanent employees receive 24 working days of paid annual leave per calendar year.

Probation employees
Employees on probation may use no more than 6 working days before confirmation.

Old policy - superseded
Before 1 January 2026, permanent employees received 21 working days of paid annual leave.
"""

    chunks = chunk_text(
        sample_text
    )

    for i, chunk in enumerate(
        chunks,
        1,
    ):
        print(
            f"\nChunk {i}:"
        )
        print(chunk)