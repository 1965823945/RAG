"""Main entry point for the RAG demo."""
from rag_minimal.llm import SimpleLLM
from rag_minimal.embeddings import FakeEmbeddings
from rag_minimal.loader import load_pdfs
from rag_minimal.chunker import chunk_documents
from rag_minimal.vectorstore import create_vector_store, load_vector_store
from rag_minimal.retriever import SimpleRetriever
from rag_minimal.chain import create_rag_chain, invoke_rag_chain


def main():
    print("=" * 60)
    print("Minimal RAG Demo")
    print("=" * 60)
    
    pdf_dir = "docs/pdfs"
    vector_store_dir = "chroma_db"
    
    print("\n[1/5] Loading PDFs...")
    docs = load_pdfs(pdf_dir)
    print(f"Loaded {len(docs)} document pages")
    
    if not docs:
        print("No PDFs found. Please add PDFs to docs/pdfs or run generate_samples.py")
        return
    
    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("\n[3/5] Creating vector store...")
    embeddings = FakeEmbeddings()
    vector_store = create_vector_store(chunks, persist_directory=vector_store_dir, embeddings=embeddings)
    print(f"Vector store created at {vector_store_dir}")
    
    print("\n[4/5] Setting up RAG chain...")
    retriever = SimpleRetriever(vector_store=vector_store, k=4)
    llm = SimpleLLM()
    chain = create_rag_chain(llm=llm, retriever=retriever)
    print("RAG chain ready!")
    
    print("\n[5/5] Demo Q&A")
    print("-" * 40)
    
    questions = [
        "What is RAG?",
        "How does LangChain work?",
        "What are vector databases?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = invoke_rag_chain(chain, question)
        print(f"Answer: {answer}")
        print("-" * 40)
    
    print("\nDemo complete!")
    print(f"\nTo run the Streamlit UI: streamlit run rag_minimal/app.py")


if __name__ == "__main__":
    main()
