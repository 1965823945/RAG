"""PDF loader module."""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_dir: str = "docs/pdfs") -> List[Document]:
    """Load all PDFs from the specified directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of Document objects
    """
    if not os.path.exists(pdf_dir):
        return []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    docs = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    
    return docs
