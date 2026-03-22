"""Shared constants for the RAG system."""

# ─────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────

DEFAULT_RAG_PROMPT = """你是一个有用的助手。请根据以下参考文档来回答用户的问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档给出回答："""


# ─────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_EMBEDDING_DIM = 384

DEFAULT_DOCS_DIR = "docs"
DEFAULT_VECTORSTORE_DIR = "chroma_db"


# ─────────────────────────────────────────────────────────────
# Planning Agent Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_QUALITY_THRESHOLD = 0.6  # 质量阈值
DEFAULT_MAX_ITERATIONS = 3  # 最大迭代次数
DEFAULT_MAX_COT_STEPS = 10  # 思维链最大步骤
DEFAULT_MAX_SUBTASKS = 7  # 最大子任务数
