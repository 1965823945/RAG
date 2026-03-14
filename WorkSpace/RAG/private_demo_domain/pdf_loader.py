try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def load_pdf_text(pdf_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is not installed.")
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append((page.extract_text() or ""))
    return "\n".join(text_parts)
