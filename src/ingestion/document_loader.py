from pathlib import Path


def load_documents(folder_path):
    folder = Path(folder_path)

    documents = []

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


if __name__ == "__main__":
    docs = load_documents("data/sample")

    print(f"Documents loaded: {len(docs)}")

    for doc in docs:
        print("\n--------------------")
        print("FILE:", doc["filename"])
        print(doc["text"])