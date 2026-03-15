"""Retrieval module."""
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma


class SimpleRetriever(BaseRetriever):
    """Simple retriever that wraps a Chroma vector store.
    
    For demonstration purposes - retrieves relevant documents based on similarity.
    """
    
    vector_store: Chroma
    k: int = 4
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents for the query."""
        return self.vector_store.similarity_search(
            query=query,
            k=self.k,
        )
