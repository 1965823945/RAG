"""Document loader module - supports Word and PDF documents."""
import os
from typing import List
from langchain_core.documents import Document

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
        return []
    
    docs = []
    files = os.listdir(doc_dir)
    
    # Load Word documents (.docx)
    docx_files = [f for f in files if f.endswith('.docx')]
    for docx_file in docx_files:
        if not _DOCX_AVAILABLE:
            print("警告: 未安装 python-docx 库，跳过 .docx 文件")
            continue
        docx_path = os.path.join(doc_dir, docx_file)
        try:
            loader = Docx2txtLoader(docx_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"加载 {docx_file} 出错: {e}")
    
    # Load PDFs (.pdf) - for backward compatibility
    pdf_files = [f for f in files if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        if not _PDF_AVAILABLE:
            print("警告: 未安装 pypdf 库，跳过 .pdf 文件")
            continue
        pdf_path = os.path.join(doc_dir, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"加载 {pdf_file} 出错: {e}")
    
    return docs


# Backward compatibility alias
def load_pdfs(pdf_dir: str = "docs") -> List[Document]:
    """Load PDFs from directory (backward compatibility)."""
    return load_documents(pdf_dir)
