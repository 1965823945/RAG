"""Knowledge search tool wrapping current RAG retrieval."""

import hashlib
from typing import Any, Dict, List
from pydantic import ValidationError

from rag_minimal.schemas import (
    SearchInput,
    SearchOutput,
    SearchResultItem,
    ErrorCode,
)
from rag_minimal.tools.base import Tool
from rag_minimal.tools.logger import logged_invoke
from rag_minimal.loader import load_documents
from rag_minimal.chunker import chunk_documents
from rag_minimal.retriever import SimpleRetriever


def generate_chunk_id(content: str, source: str, index: int) -> str:
    """Generate a unique chunk ID from content hash."""
    hash_input = f"{source}:{index}:{content[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def generate_doc_id(source: str) -> str:
    """Generate a document ID from source path."""
    return hashlib.md5(source.encode()).hexdigest()[:8]


class KnowledgeSearchTool(Tool):
    """Search user-provided documents and return relevant snippets.

    This tool loads documents from a directory, chunks them,
    and performs keyword-based retrieval.
    """

    name = "knowledge_search"
    description = "Search user-provided documents and return relevant snippets."
    version = "1.1.0"
    tags = ["search", "rag", "retrieval"]
    input_schema = SearchInput
    output_schema = SearchOutput

    def __init__(self, docs_dir: str = "docs"):
        """Initialize the knowledge search tool.

        Args:
            docs_dir: Directory containing documents to search
        """
        self.docs_dir = docs_dir
        self._chunks_cache: List[Any] = []
        self._chunk_metadata: Dict[str, Dict[str, Any]] = {}

    def _load_and_chunk(self) -> List[Any]:
        """Load documents and create chunks with metadata."""
        docs = load_documents(self.docs_dir)
        chunks = chunk_documents(docs)

        # Build metadata index
        self._chunk_metadata.clear()
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "") if chunk.metadata else ""
            doc_id = generate_doc_id(source)
            chunk_id = generate_chunk_id(chunk.page_content, source, i)

            self._chunk_metadata[id(chunk)] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "source": source,
            }

        self._chunks_cache = chunks
        return chunks

    @logged_invoke
    def invoke(self, payload: Dict[str, Any]) -> SearchOutput:
        """Execute knowledge search.

        Args:
            payload: Dict with 'query' and optional 'top_k'

        Returns:
            SearchOutput with results and tracing info
        """
        # Validate input
        try:
            data = SearchInput(**payload)
        except ValidationError as e:
            return SearchOutput(
                success=False,
                error_code=ErrorCode.INVALID_INPUT,
                message=str(e),
                query=payload.get("query", ""),
            )

        # Load and chunk documents
        try:
            chunks = self._load_and_chunk()
        except Exception as e:
            return SearchOutput(
                success=False,
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                message=f"Failed to load documents: {str(e)}",
                query=data.query,
            )

        if not chunks:
            return SearchOutput(
                success=True,
                error_code=ErrorCode.NO_RESULTS,
                message="No documents found in the specified directory",
                query=data.query,
                results=[],
                total_chunks=0,
            )

        # Perform retrieval
        retriever = SimpleRetriever(documents=chunks, k=data.top_k)
        matched_docs = retriever._get_relevant_documents(data.query)

        # Build results with tracing info
        items = []
        for doc in matched_docs:
            meta = self._chunk_metadata.get(id(doc), {})
            source = doc.metadata.get("source", "") if doc.metadata else ""

            items.append(
                SearchResultItem(
                    content=doc.page_content,
                    source=str(source),
                    score=0.0,  # TODO: Add actual scores from retriever
                    doc_id=meta.get("doc_id"),
                    chunk_id=meta.get("chunk_id"),
                    chunk_index=meta.get("chunk_index"),
                )
            )

        return SearchOutput(
            success=True,
            error_code=ErrorCode.OK,
            message="ok",
            query=data.query,
            results=items,
            total_chunks=len(chunks),
        )
