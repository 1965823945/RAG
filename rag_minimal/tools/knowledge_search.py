"""Knowledge search tool wrapping current RAG retrieval."""

from typing import Any, Dict
from pydantic import ValidationError

from rag_minimal.schemas import SearchInput, SearchOutput, SearchResultItem
from rag_minimal.tools.base import Tool
from rag_minimal.loader import load_documents
from rag_minimal.chunker import chunk_documents
from rag_minimal.retriever import SimpleRetriever


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = "Search user-provided documents and return relevant snippets."
    input_schema = SearchInput
    output_schema = SearchOutput

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir

    def invoke(self, payload: Dict[str, Any]) -> SearchOutput:
        try:
            data = self.input_schema(**payload)
        except ValidationError as e:
            return SearchOutput(
                success=False, message=str(e), query=payload.get("query", "")
            )

        docs = load_documents(self.docs_dir)
        chunks = chunk_documents(docs)
        retriever = SimpleRetriever(documents=chunks, k=data.top_k)
        matched_docs = retriever._get_relevant_documents(data.query, run_manager=None)

        items = []
        for doc in matched_docs:
            items.append(
                SearchResultItem(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "")) if doc.metadata else "",
                    score=0.0,
                )
            )

        return SearchOutput(success=True, query=data.query, results=items)
