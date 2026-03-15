"""Main entry point for the RAG demo."""
from rag_minimal.llm import SimpleLLM
from rag_minimal.embeddings import FakeEmbeddings
from rag_minimal.loader import load_documents
from rag_minimal.chunker import chunk_documents
from rag_minimal.vectorstore import create_vector_store, load_vector_store
from rag_minimal.retriever import SimpleRetriever
from rag_minimal.chain import create_rag_chain, invoke_rag_chain


def main():
    print("=" * 60)
    print("RAG 检索增强生成演示")
    print("=" * 60)
    
    doc_dir = "docs"
    vector_store_dir = "chroma_db"
    
    print("\n[1/5] 正在加载文档...")
    docs = load_documents(doc_dir)
    print(f"已加载 {len(docs)} 个文档页面")
    
    if not docs:
        print("未找到文档！请在 docs 目录下放置 .docx 或 .pdf 文件")
        print("支持的格式: .docx, .pdf")
        return
    
    print("\n[2/5] 正在分块文档...")
    chunks = chunk_documents(docs)
    print(f"已创建 {len(chunks)} 个文档块")
    
    print("\n[3/5] 正在创建向量数据库...")
    embeddings = FakeEmbeddings()
    vector_store = create_vector_store(chunks, persist_directory=vector_store_dir, embeddings=embeddings)
    print(f"向量数据库已创建: {vector_store_dir}")
    
    print("\n[4/5] 正在构建 RAG 链...")
    retriever = SimpleRetriever(vector_store=vector_store, k=3)
    llm = SimpleLLM()
    chain = create_rag_chain(llm=llm, retriever=retriever)
    print("RAG 链准备就绪！")
    
    print("\n[5/5] 演示问答")
    print("-" * 40)
    
    questions = [
        "什么是 RAG?",
        "LangChain 是什么?",
        "向量数据库是什么?",
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = invoke_rag_chain(chain, question)
        print(f"回答: {answer}")
        print("-" * 40)
    
    print("\n演示完成!")
    print(f"\n启动 Streamlit Web 界面: streamlit run rag_minimal/app.py")


if __name__ == "__main__":
    main()
