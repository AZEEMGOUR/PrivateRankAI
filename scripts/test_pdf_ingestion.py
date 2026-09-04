import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdf_chunks


PDF_FOLDER = PROJECT_ROOT / "data" / "real_documents"


def main():
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:")
        print(PDF_FOLDER)
        print("\nAdd a PDF and run again.")
        return

    for pdf_file in pdf_files:
        print("\n==============================")
        print("PDF:", pdf_file.name)
        print("==============================")

        chunks = load_pdf_chunks(pdf_file)

        print("Total chunks:", len(chunks))

        for chunk in chunks[:5]:
            print("\n------------------------------")
            print("Filename:", chunk["filename"])
            print("Page:", chunk["page"])
            print("Chunk ID:", chunk["chunk_id"])
            print("Text:")
            print(chunk["text"][:500])


if __name__ == "__main__":
    main()