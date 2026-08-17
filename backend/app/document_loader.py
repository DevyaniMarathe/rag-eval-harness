from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    """

    reader = PdfReader(pdf_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


if __name__ == "__main__":
    pdf_folder = Path("data/documents")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
    else:
        pdf_path = pdf_files[0]

        print(f"Reading: {pdf_path}")

        extracted_text = extract_text_from_pdf(str(pdf_path))

        print("\n--- EXTRACTED TEXT ---\n")
        print(extracted_text[:5000])

        print("\n--- END ---")
        print(f"\nTotal characters extracted: {len(extracted_text)}")