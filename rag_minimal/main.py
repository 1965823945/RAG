"""Main entry point for the RAG demo - using AgentRuntime."""

import os
import shutil

from rag_minimal.agent_runtime import AgentRuntime
from rag_minimal.llm import SimpleLLM
from rag_minimal.embeddings import FakeEmbeddings
from rag_minimal.loader import load_documents
from rag_minimal.chunker import chunk_documents
from rag_minimal.vectorstore import create_vector_store


def main():
    print("=" * 60)
    print("RAG 检索增强生成演示 (AgentRuntime)")
    print("=" * 60)

    doc_dir = "docs"
    vector_store_dir = "chroma_db"

    print("\n[1/4] 正在加载文档...")
    docs = load_documents(doc_dir)
    print(f"已加载 {len(docs)} 个文档页面")

    if not docs:
        print("未找到文档！请在 docs 目录下放置 .docx 或 .pdf 文件")
        print("支持的格式: .docx, .pdf")
        return

    print("\n[2/4] 正在分块文档...")
    chunks = chunk_documents(docs)
    print(f"已创建 {len(chunks)} 个文档块")

    # 尝试删除旧的向量数据库
    try:
        if os.path.exists(vector_store_dir):
            shutil.rmtree(vector_store_dir)
    except PermissionError:
        print(f"警告: 无法删除旧的 {vector_store_dir}，将创建新的向量库")
        vector_store_dir = "chroma_db_new"

    print("\n[3/4] 正在创建向量数据库...")
    embeddings = FakeEmbeddings()
    create_vector_store(
        chunks, persist_directory=vector_store_dir, embeddings=embeddings
    )
    print(f"向量数据库已创建: {vector_store_dir}")

    print("\n[4/4] 初始化 AgentRuntime...")
    llm = SimpleLLM()
    agent = AgentRuntime(docs_dir=doc_dir, llm=llm)
    print("AgentRuntime 准备就绪！")

    print("\n" + "=" * 60)
    print("演示问答")
    print("=" * 60)

    questions = [
        "什么是 RAG?",
        "LangChain 是什么?",
        "向量数据库是什么?",
    ]

    for question in questions:
        print(f"\n问题: {question}")
        result = agent.ask(question, top_k=3)
        print(f"回答: {result.answer}")
        if result.sources:
            print(f"来源: {len(result.sources)} 个文档片段")
        print("-" * 40)

    print("\n演示完成!")
    print(f"\n启动 Streamlit Web 界面: streamlit run rag_minimal/app.py")


if __name__ == "__main__":
    main()
