"""Embeddings module for RAG demo."""
from typing import List
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings for demonstration purposes.
    
    This is NOT suitable for production use - it returns constant vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        return [[0.01 * (i + 1) for _ in range(self._dimension)] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Return fake embedding for query."""
        return [0.01 for _ in range(self._dimension)]
