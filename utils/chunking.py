from unstructured.partition.pdf import partition_pdf
import fitz  # PyMuPDF for table extraction

def chunk_pdf(pdf_path, doc_type="manual"):
    """
    Custom chunking based on doc_type:
    - research: semantic chunking
    - invoice: table extraction
    - manual: section-based chunking
    """
    chunks = []

    if doc_type == "research":
        # Semantic chunking using unstructured partition, OCR disabled
        elements = partition_pdf(pdf_path, ocr_agent=None, strategy="hi_res")
        for el in elements:
            text = el.text.strip()
            if text:
                chunks.append(text)

    elif doc_type == "invoice":
        # Table extraction using PyMuPDF
        doc = fitz.open(pdf_path)
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                if text:
                    chunks.append(text)

    else:  # manual
        # Section-based chunking, OCR disabled
        elements = partition_pdf(pdf_path, ocr_agent=None, strategy="hi_res")
        current_section = ""
        for el in elements:
            text = el.text.strip()
            if text:
                if text.endswith(":"):  # section header
                    if current_section:
                        chunks.append(current_section.strip())
                    current_section = text
                else:
                    current_section += " " + text
        if current_section:
            chunks.append(current_section.strip())

    return chunks
