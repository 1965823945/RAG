"""Text chunking module."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into smaller chunks.
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)
