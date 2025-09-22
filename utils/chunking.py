# chunking.py
import os
from unstructured.partition.pdf import partition_pdf
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO)

def chunk_pdf(pdf_path, doc_type="manual"):
    """
    Custom chunking based on doc_type:
    - research: semantic chunking
    - invoice: table extraction
    - manual: section-based chunking
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chunks = []

    try:
        if doc_type == "research":
            elements = partition_pdf(pdf_path)
            for el in elements:
                if hasattr(el, "text") and el.text:
                    text = el.text.strip()
                    if text:
                        chunks.append(text)

        elif doc_type == "invoice":
            doc = fitz.open(pdf_path)
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    if len(b) > 4 and b[4]:
                        text = b[4].strip()
                        if text:
                            chunks.append(text)

        else:  # manual
            elements = partition_pdf(pdf_path)
            current_section = ""
            for el in elements:
                if hasattr(el, "text") and el.text:
                    text = el.text.strip()
                    if text.endswith(":"):  # section header
                        if current_section:
                            chunks.append(current_section.strip())
                        current_section = text
                    else:
                        current_section += " " + text
            if current_section:
                chunks.append(current_section.strip())

    except Exception as e:
        logging.error(f"Error during chunking PDF {pdf_path}: {e}")
        raise RuntimeError(f"Error during chunking: {e}")

    return chunks
