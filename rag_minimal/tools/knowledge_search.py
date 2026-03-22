"""Knowledge search tool wrapping current RAG retrieval."""

import os
import hashlib
import logging
from typing import Any, Dict, List, Optional
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

logger = logging.getLogger("rag_minimal.tools")


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

    Features:
    - Lazy loading: documents are loaded only on first search
    - Caching: chunks are cached until refresh() is called
    - Auto-refresh: detects directory modification time changes
    """

    name = "knowledge_search"
    description = "Search user-provided documents and return relevant snippets."
    version = "1.2.0"
    tags = ["search", "rag", "retrieval"]
    input_schema = SearchInput
    output_schema = SearchOutput

    def __init__(self, docs_dir: str = "docs", auto_refresh: bool = True):
        """Initialize the knowledge search tool.

        Args:
            docs_dir: Directory containing documents to search
            auto_refresh: If True, auto-refresh when directory changes
        """
        self.docs_dir = docs_dir
        self.auto_refresh = auto_refresh

        # Cache
        self._chunks: Optional[List[Any]] = None
        self._chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self._last_mtime: float = 0.0

    def _get_dir_mtime(self) -> float:
        """Get the latest modification time of the docs directory."""
        if not os.path.exists(self.docs_dir):
            return 0.0

        latest = os.path.getmtime(self.docs_dir)
        try:
            for entry in os.scandir(self.docs_dir):
                if entry.is_file():
                    latest = max(latest, entry.stat().st_mtime)
        except OSError:
            pass
        return latest

    def _needs_refresh(self) -> bool:
        """Check if documents need to be reloaded."""
        if self._chunks is None:
            return True
        if not self.auto_refresh:
            return False
        return self._get_dir_mtime() > self._last_mtime

    def refresh(self) -> int:
        """Force reload documents from disk.

        Returns:
            Number of chunks loaded
        """
        logger.info(f"Loading documents from {self.docs_dir}")

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

        self._chunks = chunks
        self._last_mtime = self._get_dir_mtime()

        logger.info(f"Loaded {len(chunks)} chunks from {len(docs)} documents")
        return len(chunks)

    def _ensure_loaded(self) -> List[Any]:
        """Ensure documents are loaded, refresh if needed."""
        if self._needs_refresh():
            self.refresh()
        return self._chunks or []

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

        # Load documents (with caching)
        try:
            chunks = self._ensure_loaded()
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
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
                    score=0.0,
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
