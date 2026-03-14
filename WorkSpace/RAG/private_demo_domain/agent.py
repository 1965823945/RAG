from pdf_loader import load_pdf_text
from chunker import chunk_text
from embeddings import embed_texts
from vector_store import VectorStore
from llm import generate_from_context

class QAAgent:
    def __init__(self, pdf_path: str, dim: int = 128, k: int = 3):
        self.pdf_path = pdf_path
        self.dim = dim
        self.k = k
        self.store = VectorStore(dim=dim)
        self._ingest()

    def _ingest(self):
        text = load_pdf_text(self.pdf_path) if self.pdf_path else ""
        if not text:
            return
        chunks = chunk_text(text, max_len=600, overlap=100)
        self.store.add_documents(chunks)

    def answer(self, question: str) -> str:
        top_docs = self.store.search(question, top_k=self.k)
        context = "\n\n".join([d["text"] for d in top_docs]) if top_docs else ""
        return generate_from_context(context, question)
