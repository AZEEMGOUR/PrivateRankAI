def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_text = """
    Employees are entitled to 24 paid leaves per calendar year.
    Leave requests must be approved by the reporting manager.
    Unused paid leaves may be carried forward according to company policy.
    """

    chunks = chunk_text(sample_text)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(chunk)