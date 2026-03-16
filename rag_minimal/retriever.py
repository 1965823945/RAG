"""Simple keyword-based retriever for demo purposes."""

import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma


class SimpleRetriever(BaseRetriever):
    """Keyword-based retriever for demo - more reliable than fake embeddings."""

    vector_store: Chroma
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve documents using keyword matching."""

        # Get all documents from vector store
        try:
            result = self.vector_store.get()
            all_docs = result.get("documents", [])
            all_metadatas = result.get("metadatas", [])
        except Exception:
            # Fallback to similarity search
            return self.vector_store.similarity_search(query=query, k=self.k)

        if not all_docs:
            return []

        # Extract keywords from query (Chinese + English)
        query_lower = query.lower()
        # Extract Chinese characters and English words
        query_terms = set(re.findall(r"[\w]+", query_lower))

        # Score each document by keyword overlap
        scored = []
        for i, doc_text in enumerate(all_docs):
            doc_lower = doc_text.lower()
            doc_terms = set(re.findall(r"[\w]+", doc_lower))

            # Calculate overlap
            overlap = query_terms & doc_terms
            score = len(overlap)

            # Boost if query appears at start of document
            first_chars = doc_lower[:200]
            for term in overlap:
                if term in first_chars:
                    score += 3

            # Extra boost for exact phrase match
            if query_lower in doc_lower:
                score += 10

            # Boost for title-like content (first line)
            first_line = doc_text.split("\n")[0] if "\n" in doc_text else doc_text[:50]
            if any(term in first_line.lower() for term in query_terms if len(term) > 1):
                score += 2

            if score > 0:
                metadata = all_metadatas[i] if i < len(all_metadatas) else {}
                scored.append((score, i, doc_text, metadata))

        # Sort by score
        scored.sort(key=lambda x: -x[0])

        # Return top k
        results = []
        for score, i, text, metadata in scored[: self.k]:
            doc = Document(page_content=text, metadata=metadata)
            results.append(doc)

        return results
