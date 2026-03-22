"""Document loader module - supports Word and PDF documents."""

import os
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("rag_minimal")

# Try to import loaders
try:
    from langchain_community.document_loaders import PyPDFLoader

    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import Docx2txtLoader

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


def load_documents(doc_dir: str = "docs") -> List[Document]:
    """Load all documents (Word and PDF) from the specified directory.

    Args:
        doc_dir: Directory containing document files

    Returns:
        List of Document objects
    """
    if not os.path.exists(doc_dir):
        logger.warning(f"Document directory not found: {doc_dir}")
        return []

    docs = []
    files = os.listdir(doc_dir)

    # Load Word documents (.docx)
    docx_files = [f for f in files if f.endswith(".docx")]
    for docx_file in docx_files:
        if not _DOCX_AVAILABLE:
            logger.warning("python-docx not installed, skipping .docx files")
            continue
        docx_path = os.path.join(doc_dir, docx_file)
        try:
            loader = Docx2txtLoader(docx_path)
            docs.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load {docx_file}: {e}")

    # Load PDFs (.pdf)
    pdf_files = [f for f in files if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        if not _PDF_AVAILABLE:
            logger.warning("pypdf not installed, skipping .pdf files")
            continue
        pdf_path = os.path.join(doc_dir, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}: {e}")

    logger.info(f"Loaded {len(docs)} documents from {doc_dir}")
    return docs


# Backward compatibility alias
def load_pdfs(pdf_dir: str = "docs") -> List[Document]:
    """Load PDFs from directory (backward compatibility)."""
    return load_documents(pdf_dir)
