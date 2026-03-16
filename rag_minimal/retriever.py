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
        if documents is None and "vector_store" in kwargs:
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

        # Clean and prepare query
        query_lower = query.lower().strip()

        # Score each document
        scored = []
        for i, doc in enumerate(self.documents):
            doc_text = doc.page_content.lower()
            doc_title = doc.metadata.get("source", "") if doc.metadata else ""
            if doc_title:
                doc_title = doc_title.lower()

            score = 0

            # Check if query appears as substring (most important)
            if query_lower in doc_text:
                score += 100

            # Check for partial matches in title
            if doc_title and query_lower in doc_title:
                score += 50

            # Check each word in query
            words = re.findall(r"[\w\u4e00-\u9fff]+", query_lower)
            for word in words:
                if len(word) < 2:  # Skip single characters
                    continue
                # Count occurrences in document
                count = doc_text.count(word)
                if count > 0:
                    score += count * 2
                # Extra points if word is in first 200 chars
                if word in doc_text[:200]:
                    score += 5
                # Extra points if word is in title
                if doc_title and word in doc_title:
                    score += 10

            if score > 0:
                scored.append((score, i, doc))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Return top k results
        return [doc for score, i, doc in scored[: self.k]]
