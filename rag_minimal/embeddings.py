"""Embeddings module for RAG demo."""

import hashlib
from typing import List
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings for demonstration purposes.

    This is NOT suitable for production use - it returns deterministic vectors based on text content.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def _get_deterministic_vector(self, text: str) -> List[float]:
        """Generate a deterministic vector from text."""
        # Create a hash of the text
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Convert hash to a list of floats
        return [float(b) / 255.0 for b in hash_bytes] * (self._dimension // 16 + 1)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        return [
            self._get_deterministic_vector(text)[: self._dimension] for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return fake embedding for query."""
        return self._get_deterministic_vector(text)[: self._dimension]
