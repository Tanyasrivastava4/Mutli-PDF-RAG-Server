import os
from unstructured.partition.pdf import partition_pdf
import fitz  # PyMuPDF for table extraction

# Folder where all PDFs are stored
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # ensure folder exists

def chunk_pdf(file_path, doc_type="manual"):
    """
    Custom chunking based on doc_type:
    - research: semantic chunking
    - invoice: table extraction
    - manual: section-based chunking
    OCR is completely disabled.
    """

    # Ensure file is inside UPLOAD_DIR
    if not file_path.startswith(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(file_path))

    chunks = []

    if doc_type == "research":
        # Semantic chunking using unstructured partition, OCR disabled
        elements = partition_pdf(file_path, strategy="fast")
        for el in elements:
            text = getattr(el, "text", None)
            if text:
                chunks.append(text.strip())

    elif doc_type == "invoice":
        # Table extraction using PyMuPDF
        doc = fitz.open(file_path)
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                if text:
                    chunks.append(text)

    else:  # manual
        # Section-based chunking using unstructured partition, OCR disabled
        elements = partition_pdf(file_path, strategy="fast")
        current_section = ""
        for el in elements:
            text = getattr(el, "text", None)
            if text:
                text = text.strip()
                if text.endswith(":"):  # section header
                    if current_section:
                        chunks.append(current_section.strip())
                    current_section = text
                else:
                    current_section += " " + text
        if current_section:
            chunks.append(current_section.strip())

    return chunks
