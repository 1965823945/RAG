"""Improved retriever with keyword-based reranking."""

import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma


class SimpleRetriever(BaseRetriever):
    """Improved retriever with keyword reranking for demo."""

    vector_store: Chroma
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve and rerank documents."""
        # Get more candidates than needed
        candidates = self.vector_store.similarity_search(query=query, k=self.k * 3)

        if not candidates:
            return []

        # Rerank by keyword matching
        reranked = self._rerank(query, candidates)

        return reranked[: self.k]

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on keyword overlap with query."""
        # Extract keywords from query
        query_lower = query.lower()
        # Extract Chinese and English words
        query_words = set(re.findall(r"[\w\u4e00-\u9fff]+", query_lower))

        # Score each document
        scored = []
        for doc in documents:
            doc_text = doc.page_content.lower()
            doc_words = set(re.findall(r"[\w\u4e00-\u9fff]+", doc_text))

            # Calculate keyword overlap
            overlap = query_words & doc_words

            # Boost score for exact matches at start of document
            boost = 0
            for word in overlap:
                if word in doc_text[:100]:  # First 100 chars
                    boost += 2

            score = len(overlap) + boost
            scored.append((score, doc))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Return reranked documents
        return [doc for score, doc in scored]
