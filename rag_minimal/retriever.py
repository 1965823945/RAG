"""Keyword-based retriever for better demo results."""

import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class SimpleRetriever(BaseRetriever):
    """Keyword-based retriever for demo - more reliable than vector search with fake embeddings."""

    documents: List[Document]
    k: int = 3

    def __init__(self, documents: List[Document] = None, k: int = 3, **kwargs):
        # Accept either documents or vector_store for compatibility
        if documents is None and "vector_store" in kwargs:
            # Fallback to old behavior
            vector_store = kwargs.pop("vector_store")
            self.vector_store = vector_store
            documents = []

        super().__init__(documents=documents or [], k=k, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents using keyword matching."""
        if not self.documents:
            return []

        # Extract query keywords
        query_lower = query.lower()
        query_words = set(re.findall(r"[\w\u4e00-\u9fff]+", query_lower))

        # Score each document
        scored = []
        for i, doc in enumerate(self.documents):
            doc_text = doc.page_content.lower()
            doc_words = set(re.findall(r"[\w\u4e00-\u9fff]+", doc_text))

            # Calculate keyword overlap
            overlap = query_words & doc_words

            if not overlap:
                continue

            # Calculate score
            score = len(overlap)

            # Boost for exact phrase match
            if query_lower in doc_text:
                score += 10

            # Boost for match in first 200 chars
            first_200 = doc_text[:200]
            for word in overlap:
                if word in first_200:
                    score += 2

            # Boost for shorter documents (more likely to be relevant)
            if len(doc.page_content) < 400:
                score *= 1.5

            scored.append((score, i, doc))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Return top k results
        return [doc for score, i, doc in scored[: self.k]]
