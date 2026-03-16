"""Retrieval module with keyword fallback."""

import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma


class SimpleRetriever(BaseRetriever):
    """Simple retriever with keyword fallback for better demo results."""

    vector_store: Chroma
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents using similarity + keyword matching."""
        # First try vector similarity
        try:
            docs = self.vector_store.similarity_search(query=query, k=self.k)
            if docs:
                # Also do keyword matching as backup
                keyword_docs = self._keyword_search(query, k=2)
                # Combine results, preferring vector results
                seen = set(d.page_content[:50] for d in docs)
                for doc in keyword_docs:
                    if len(docs) >= self.k:
                        break
                    key = doc.page_content[:50]
                    if key not in seen:
                        docs.append(doc)
                        seen.add(key)
                return docs
        except Exception:
            pass

        # Fallback to keyword search
        return self._keyword_search(query, k=self.k)

    def _keyword_search(self, query: str, k: int = 3) -> List[Document]:
        """Simple keyword-based search as fallback."""
        query_words = set(re.findall(r"\w+", query.lower()))

        # Get all docs from vector store
        try:
            all_docs = self.vector_store.get()["documents"]
            if not all_docs:
                return []

            # Score by keyword overlap
            scored = []
            for i, doc_text in enumerate(all_docs):
                doc_words = set(re.findall(r"\w+", doc_text.lower()))
                overlap = len(query_words & doc_words)
                if overlap > 0:
                    # Also check if query words appear in document title/start
                    score = overlap * 2
                    if any(qw in doc_text[:200].lower() for qw in query_words):
                        score += 3
                    scored.append((score, i, doc_text))

            # Sort by score and return top k
            scored.sort(key=lambda x: -x[0])
            results = []
            for score, i, text in scored[:k]:
                doc = Document(page_content=text, metadata={"index": i})
                results.append(doc)

            return results
        except Exception:
            return []
