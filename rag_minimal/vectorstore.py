"""Vector store module using ChromaDB."""

import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from rag_minimal.embeddings import FakeEmbeddings

logger = logging.getLogger(__name__)

# Try to use new langchain_chroma, fall back to old Chroma
try:
    from langchain_chroma import Chroma as ChromaDB
except ImportError:
    from langchain_community.vectorstores import Chroma as ChromaDB


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class VectorStoreCreateError(VectorStoreError):
    """Error creating vector store."""

    pass


class VectorStoreLoadError(VectorStoreError):
    """Error loading vector store."""

    pass


def create_vector_store(
    documents: List[Document],
    persist_directory: str = "chroma_db",
    embeddings: Optional[FakeEmbeddings] = None,
) -> ChromaDB:
    """Create a Chroma vector store from documents.

    Args:
        documents: List of documents to index
        persist_directory: Directory to persist the vector store
        embeddings: Embeddings model (defaults to FakeEmbeddings)

    Returns:
        ChromaDB vector store instance

    Raises:
        VectorStoreCreateError: If creation fails
    """
    if not documents:
        raise VectorStoreCreateError("No documents provided to create vector store")

    if embeddings is None:
        embeddings = FakeEmbeddings()

    try:
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(
            f"Creating vector store with {len(documents)} documents in {persist_directory}"
        )

        vector_store = ChromaDB.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        logger.info("Vector store created successfully")
        return vector_store

    except PermissionError as e:
        raise VectorStoreCreateError(
            f"Permission denied for directory {persist_directory}: {e}"
        )
    except Exception as e:
        raise VectorStoreCreateError(f"Failed to create vector store: {e}")


def load_vector_store(
    persist_directory: str = "chroma_db", embeddings: Optional[FakeEmbeddings] = None
) -> Optional[ChromaDB]:
    """Load an existing Chroma vector store.

    Args:
        persist_directory: Directory where vector store is persisted
        embeddings: Embeddings model (defaults to FakeEmbeddings)

    Returns:
        ChromaDB instance or None if directory doesn't exist

    Raises:
        VectorStoreLoadError: If loading fails
    """
    if not os.path.exists(persist_directory):
        logger.debug(f"Vector store directory not found: {persist_directory}")
        return None

    if embeddings is None:
        embeddings = FakeEmbeddings()

    try:
        logger.info(f"Loading vector store from {persist_directory}")
        store = ChromaDB(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        logger.info("Vector store loaded successfully")
        return store

    except PermissionError as e:
        raise VectorStoreLoadError(
            f"Permission denied for directory {persist_directory}: {e}"
        )
    except Exception as e:
        raise VectorStoreLoadError(f"Failed to load vector store: {e}")
