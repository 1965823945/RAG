"""Day 2: Load PDFs, chunk, embed, and store in ChromaDB."""
import os
from pathlib import Path
from glob import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma


CHROMA_DIR = "chromadb"
PDFS_DIR = "docs/pdfs"

# Compatibility: prefer langchain community Chroma if available
try:
    from langchain_community.vectorstores import Chroma as CommunityChroma
    _CHROMA_USE_COMMUNITY = True
except Exception:
    _CHROMA_USE_COMMUNITY = False
    from langchain.vectorstores import Chroma as CommunityChroma


def ensure_dirs():
    Path(PDFS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


def generate_sample_pdfs(n: int = 10):
    """Generate sample PDFs if none exist (requires reportlab)."""
    try:
        from reportlab.pdfgen import canvas
    except Exception:
        print("reportlab not installed; cannot generate sample PDFs automatically.")
        return
    ensure_dirs()
    for i in range(1, n + 1):
        c = canvas.Canvas(f"{PDFS_DIR}/sample_{i}.pdf")
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, f"Sample PDF {i}")
        c.drawString(100, 730, "This is a minimal document for RAG demo.")
        c.save()


def load_documents_from_pdfs():
    docs = []
    pdf_files = glob(os.path.join(PDFS_DIR, "*.pdf"))
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load_and_split())
    return docs


def build_vector_store(persist_dir: str = CHROMA_DIR) -> Chroma:
    ensure_dirs()
    # If no PDFs exist, try to generate some samples
    if not glob(os.path.join(PDFS_DIR, "*.pdf")):
        generate_sample_pdfs(10)
    docs = load_documents_from_pdfs()
    if not docs:
        print("No documents loaded. Ensure PDFs exist in docs/pdfs.")
        return None
    embeddings = FakeEmbeddings(size=1536)
    # Instantiate vector store with compatibility across LangChain versions
    if _CHROMA_USE_COMMUNITY:
        try:
            vectordb = CommunityChroma(persist_directory=persist_dir, embedding=embeddings)
        except TypeError:
            vectordb = CommunityChroma(persist_directory=persist_dir, embeddings=embeddings)
    else:
        vectordb = CommunityChroma(persist_directory=persist_dir, embedding=embeddings)
    vectordb.add_documents(docs)
    vectordb.persist()
    return vectordb


if __name__ == "__main__":
    build_vector_store()
