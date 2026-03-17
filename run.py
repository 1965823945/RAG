"""
RAG 启动器 - 一键启动 RAG 系统
"""

import os
import sys

# 添加当前目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import shutil
import subprocess
from pathlib import Path


def check_dependencies():
    """检查依赖是否安装"""
    required = ["streamlit", "langchain", "chromadb"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    return True


def rebuild_vector_store():
    """重建向量数据库"""
    # Always delete old database before creating new one
    vector_store_dir = "chroma_db"
    if os.path.exists(vector_store_dir):
        print(f"正在删除旧的向量数据库...")
        import time

        time.sleep(1)  # Wait a bit
        try:
            shutil.rmtree(vector_store_dir)
        except Exception as e:
            print(f"无法删除旧数据库，将使用新名称: {e}")
            # Use a new name instead
            vector_store_dir = f"chroma_db_{int(time.time())}"

    print(f"\n正在加载文档并创建向量数据库...")

    # Import and run
    from rag_minimal.loader import load_documents
    from rag_minimal.chunker import chunk_documents
    from rag_minimal.vectorstore import create_vector_store
    from rag_minimal.embeddings import FakeEmbeddings

    docs = load_documents("docs")
    if not docs:
        print("错误: 未找到文档！请在 docs 目录下放置 .docx 或 .pdf 文件")
        return False

    print(f"已加载 {len(docs)} 个文档页面")

    chunks = chunk_documents(docs)
    print(f"已创建 {len(chunks)} 个文档块")

    embeddings = FakeEmbeddings()
    create_vector_store(
        chunks, persist_directory=vector_store_dir, embeddings=embeddings
    )
    print(f"向量数据库已创建: {vector_store_dir}")

    return True


def main():
    print("=" * 60)
    print("       RAG 检索增强生成系统")
    print("=" * 60)
    print()

    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return

    # 检查文档目录
    if not os.path.exists("docs"):
        print("错误: docs 目录不存在！")
        print("请创建 docs 目录并放置 .docx 或 .pdf 文件")
        input("\n按回车键退出...")
        return

    # 检查是否有文档
    doc_files = []
    for ext in ["*.docx", "*.pdf"]:
        doc_files.extend(list(Path("docs").glob(ext)))

    if not doc_files:
        print("错误: docs 目录下没有文档文件！")
        print("请放置 .docx 或 .pdf 文件")
        input("\n按回车键退出...")
        return

    print(f"找到 {len(doc_files)} 个文档文件")
    print()

    # 选择操作
    print("请选择操作:")
    print("  1. 重新加载文档库（重建向量数据库）")
    print("  2. 直接启动（使用现有向量数据库）")
    print("  3. 退出")
    print()

    choice = input("请输入选项 (1/2/3): ").strip()
    print()

    if choice == "1":
        success = rebuild_vector_store()
        if not success:
            input("\n按回车键退出...")
            return
    elif choice == "2":
        if not os.path.exists("chroma_db"):
            print("向量数据库不存在，请先选择选项 1 重建")
            input("\n按回车键退出...")
            return
        print("将使用现有的向量数据库启动")
    else:
        return

    # 启动 Streamlit
    print("\n正在启动 Web 界面...")
    print("请访问: http://localhost:8501")
    print("\n按 Ctrl+C 可以停止服务")
    print("-" * 40)

    # 使用 Popen 启动
    os.chdir(script_dir)  # 切换到脚本目录
    process = subprocess.Popen(
        ["streamlit", "run", "rag_minimal/app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
