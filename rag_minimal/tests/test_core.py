"""Tests for rag_minimal package."""
from rag_minimal.llm import SimpleLLM
from rag_minimal.embeddings import FakeEmbeddings
from rag_minimal.loader import load_pdfs
from rag_minimal.chunker import chunk_documents


class TestSimpleLLM:
    """Tests for SimpleLLM."""

    def test_generate(self):
        llm = SimpleLLM()
        result = llm.generate(["Hello"])
        assert result is not None
        assert len(result.generations) == 1


class TestFakeEmbeddings:
    """Tests for FakeEmbeddings."""

    def test_embedding_dimension(self):
        emb = FakeEmbeddings(dimension=128)
        assert emb.embedding_dimension == 128

    def test_embed_documents(self):
        emb = FakeEmbeddings(dimension=64)
        texts = ["hello", "world"]
        result = emb.embed_documents(texts)
        assert len(result) == 2
        assert len(result[0]) == 64

    def test_embed_query(self):
        emb = FakeEmbeddings(dimension=64)
        result = emb.embed_query("test query")
        assert len(result) == 64


class TestLoader:
    """Tests for PDF loader."""

    def test_load_pdfs_empty_dir(self, tmp_path):
        docs = load_pdfs(str(tmp_path))
        assert docs == []


class TestChunker:
    """Tests for text chunker."""

    def test_chunk_empty(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_with_docs(self):
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="This is a long text " * 100, metadata={"source": "test"})
        ]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0
