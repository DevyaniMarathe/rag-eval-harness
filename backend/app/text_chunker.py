from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size = 1000,
    chunk_overlap = 200
) -> list[str]:
    """
    Split extracted document text into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":
    from backend.app.document_loader import extract_text_from_pdf
    from pathlib import Path

    pdf_folder = Path("data/documents")
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
    else:
        pdf_path = pdf_files[0]

        text = extract_text_from_pdf(str(pdf_path))

        chunks = split_text(text)

        print(f"Total chunks created: {len(chunks)}")

        print("\n--- FIRST CHUNK ---\n")
        print(chunks[0])

        print("\n--- SECOND CHUNK ---\n")
        print(chunks[1])

        print("\n--- CHUNK SIZES ---")

        for i, chunk in enumerate(chunks[:10]):
            print(f"Chunk {i + 1}: {len(chunk)} characters")