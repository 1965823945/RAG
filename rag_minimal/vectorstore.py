"""Vector store module using ChromaDB."""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from rag_minimal.embeddings import FakeEmbeddings


def create_vector_store(
    documents: List[Document],
    persist_directory: str = "chroma_db",
    embeddings: Optional[FakeEmbeddings] = None
) -> Chroma:
    """Create a Chroma vector store from documents.
    
    Args:
        documents: List of Document objects
        persist_directory: Directory to persist the vector store
        embeddings: Embeddings to use (defaults to FakeEmbeddings)
        
    Returns:
        Chroma vector store
    """
    if not documents:
        raise ValueError("No documents provided to create vector store")
    
    if embeddings is None:
        embeddings = FakeEmbeddings()
    
    os.makedirs(persist_directory, exist_ok=True)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    
    return vector_store


def load_vector_store(
    persist_directory: str = "chroma_db",
    embeddings: Optional[FakeEmbeddings] = None
) -> Optional[Chroma]:
    """Load an existing Chroma vector store.
    
    Args:
        persist_directory: Directory where the vector store is persisted
        embeddings: Embeddings to use (defaults to FakeEmbeddings)
        
    Returns:
        Chroma vector store or None if not found
    """
    if not os.path.exists(persist_directory):
        return None
    
    if embeddings is None:
        embeddings = FakeEmbeddings()
    
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
