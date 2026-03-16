# RAG 检索增强生成系统

一个简洁的端到端 RAG (Retrieval-Augmented Generation) 工作流程，使用 Python、LangChain 和 ChromaDB 构建。

## 功能特点

- 支持 Word 文档 (.docx) 和 PDF 文档 (.pdf) 加载
- 文本分块和嵌入生成
- ChromaDB 向量存储
- 演示用简单 LLM（无需 API 密钥）
- Streamlit Web 界面进行交互式问答

## 项目结构

```
rag_minimal/
├── __init__.py         # 包初始化
├── main.py             # 主入口 - 运行完整演示
├── llm.py              # 简单语言模型
├── embeddings.py       # 嵌入模块
├── loader.py           # 文档加载器（支持 Word 和 PDF）
├── chunker.py          # 文本分块
├── vectorstore.py      # ChromaDB 向量存储
├── retriever.py        # 检索模块
├── chain.py            # RAG 链
├── app.py              # Streamlit Web 界面
└── tests/              # 测试文件

docs/                   # 文档库目录（放置您的 .docx 或 .pdf 文件）
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 方式一：一键启动（推荐）

```bash
python run.py
```

启动器会：
1. 检查依赖是否安装
2. 让你选择是否重建向量数据库
3. 自动打开浏览器访问 Web 界面

### 方式二：手动运行

1. **准备文档库**
   
   在 `docs/` 目录下放置您的 Word 文档 (.docx) 或 PDF 文档 (.pdf)。
   
   ```
   docs/
   ├── 公司介绍.docx
   ├── 产品手册.pdf
   └── ...
   ```

2. **运行演示**
   ```bash
   python -m rag_minimal.main
   ```

3. **启动 Web 界面**
   ```bash
   streamlit run rag_minimal/app.py
   ```

## 打包为 EXE（可选）

```bash
# 安装 PyInstaller
pip install pyinstaller

# 打包
pyinstaller build.spec

# 打包完成后在 dist/RAG_System 目录下找到 RAG_System.exe
```

## 使用说明

- 文档目录：`docs/`（可放置 .docx 或 .pdf 文件）
- 向量数据库：`chroma_db/`（首次运行自动创建）
- 检索配置：可通过侧边栏调整检索文档数量

## 运行测试

```bash
pytest --cov=rag_minimal --cov-report=xml
```

## 静态分析

```bash
ruff check rag_minimal/
```

## 依赖说明

- **langchain**: LLM 应用开发框架
- **chromadb**: 向量数据库
- **python-docx**: Word 文档支持
- **pypdf**: PDF 文档支持
- **streamlit**: Web 界面

## 注意事项

- 这是一个最小化、自包含的示例，使用本地离线组件以避免 API 密钥
- 生产环境使用时，请将 `SimpleLLM` 和 `FakeEmbeddings` 替换为真实的 LLM 提供商和嵌入模型
