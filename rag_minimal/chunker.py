"""Improved chunker with better text splitting for Chinese."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into smaller chunks with better boundaries for Chinese."""
    if not documents:
        return []

    # Use separators optimized for Chinese
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n\n",  # Multiple newlines
            "\n\n",  # Double newline
            "。\n",  # Chinese period + newline
            "。",  # Chinese period
            "！",  # Chinese exclamation
            "？",  # Chinese question
            "\n",  # Single newline
            ". ",  # English period with space
            "! ",  # English exclamation
            "? ",  # English question
            ";",  # Semicolon
            "，",  # Chinese comma
            ",",  # English comma
            " ",  # Space
            "",  # Fallback
        ],
        keep_separator=True,
    )

    chunks = text_splitter.split_documents(documents)

    # Filter out very short chunks and add metadata
    filtered_chunks = []
    for i, chunk in enumerate(chunks):
        # Skip very short chunks
        if len(chunk.page_content.strip()) < 20:
            continue

        # Add metadata
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_id"] = i

        filtered_chunks.append(chunk)

    return filtered_chunks
