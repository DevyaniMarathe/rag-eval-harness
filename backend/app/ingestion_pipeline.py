from pathlib import Path
from uuid import uuid4

from backend.app.document_loader import extract_text_from_pdf
from backend.app.text_chunker import split_text
from backend.app.embedding_service import generate_embeddings
from backend.app.qdrant_config import client


COLLECTION_NAME = "documents"


def ingest_pdf(pdf_path: str):
    print(f"Processing: {pdf_path}")

    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")

    # 2. Split text into chunks
    chunks = split_text(text)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks found. Stopping ingestion.")
        return

    # 3. Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print("Embeddings generated")

    # 4. Prepare points for Qdrant
    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            {
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "source": Path(pdf_path).name,
                    "chunk_id": index,
                },
            }
        )

    # 5. Upload points to Qdrant
    print("Uploading vectors to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Successfully uploaded {len(points)} chunks to Qdrant.")


if __name__ == "__main__":
    pdf_folder = Path("data/documents")
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/documents/")
    else:
        for pdf_path in pdf_files:
            ingest_pdf(str(pdf_path))