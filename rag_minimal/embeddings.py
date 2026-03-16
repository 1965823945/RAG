"""Embeddings module for RAG demo."""

import hashlib
from typing import List
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings for demonstration purposes.

    Uses text content to generate deterministic vectors.
    For better results, consider using real embeddings like OpenAIEmbeddings.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def _get_deterministic_vector(self, text: str) -> List[float]:
        """Generate a deterministic vector from text content."""
        # Normalize text
        text = text.lower().strip()

        # Create multiple hashes for better distribution
        hash1 = hashlib.sha256(text.encode()).digest()
        hash2 = hashlib.sha256((text + "suffix").encode()).digest()
        hash3 = hashlib.md5(text.encode()).digest()

        # Combine hashes
        combined = hash1 + hash2 + hash3

        # Convert to floats with better distribution
        vector = [float(b) / 255.0 for b in combined]

        # Repeat to match dimension
        while len(vector) < self._dimension:
            vector = vector * 2

        return vector[: self._dimension]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        return [self._get_deterministic_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return fake embedding for query."""
        return self._get_deterministic_vector(text)
