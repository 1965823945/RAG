"""Vector store module using ChromaDB."""

import os
from typing import List, Optional
from langchain_core.documents import Document
from rag_minimal.embeddings import FakeEmbeddings

# Try to use new langchain_chroma, fall back to old Chroma
try:
    from langchain_chroma import Chroma as ChromaDB
except ImportError:
    from langchain_community.vectorstores import Chroma as ChromaDB


def create_vector_store(
    documents: List[Document],
    persist_directory: str = "chroma_db",
    embeddings: Optional[FakeEmbeddings] = None,
) -> ChromaDB:
    """Create a Chroma vector store from documents."""
    if not documents:
        raise ValueError("No documents provided to create vector store")

    if embeddings is None:
        embeddings = FakeEmbeddings()

    os.makedirs(persist_directory, exist_ok=True)

    vector_store = ChromaDB.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


def load_vector_store(
    persist_directory: str = "chroma_db", embeddings: Optional[FakeEmbeddings] = None
) -> Optional[ChromaDB]:
    """Load an existing Chroma vector store."""
    if not os.path.exists(persist_directory):
        return None

    if embeddings is None:
        embeddings = FakeEmbeddings()

    return ChromaDB(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
