# RAG 检索增强生成系统

[![CI](https://github.com/1965823945/RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/1965823945/RAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个模块化、可扩展的 RAG (Retrieval-Augmented Generation) 系统，支持多工具调用、自主规划 Agent 和多轮对话记忆。

## 功能特点

### 核心功能
- 支持 Word (.docx) 和 PDF (.pdf) 文档加载
- 文本分块和向量嵌入
- ChromaDB 向量存储
- 关键词检索 + 向量检索
- Streamlit Web 界面

### 标准化工具系统
- 统一的 Tool 接口（name/description/input_schema/output_schema/invoke）
- ToolRegistry 工具注册中心
- 支持导出为 OpenAI Function Calling 和 MCP 格式
- 完整的 JSON Schema 校验（Pydantic）

### 多工具调用
- 单工具调用：`invoke_tool()`
- 并行多工具：`invoke_tools(parallel=True)`
- 顺序多工具：`invoke_tools(parallel=False)`
- 批量调用：`invoke_many()` - 同一工具不同参数
- 链式调用：`invoke_chain()` - 结果传递

### 自主规划 Agent
- **任务分解** (TaskDecomposer)：将复杂任务拆解为子任务
- **思维链** (ChainOfThought)：ReAct 模式逐步推理
- **自省反思** (SelfReflection)：质量评估和改进建议
- **规划 Agent** (PlanningAgent)：协调分解、推理、反思

### 记忆系统
- **短期记忆** (ShortTermMemory)：最近上下文，自动过期
- **长期记忆** (LongTermMemory)：持久化事实、偏好、摘要
- **工作记忆** (WorkingMemory)：当前任务上下文
- **对话管理** (ConversationManager)：多轮对话历史
- **对话 Agent** (ConversationalAgent)：集成记忆的对话 AI

### MCP Server
- JSON-RPC over stdio 协议
- 无外部 SDK 依赖
- 支持 Claude Desktop / Cursor 集成

## 项目结构

```
rag_minimal/
├── __init__.py              # 包导出
├── main.py                  # 主入口
├── app.py                   # Streamlit Web 界面
├── schemas.py               # Pydantic 数据模型
├── constants.py             # 共享常量
│
├── # 核心模块
├── llm.py                   # 简单 LLM
├── llm_config.py            # LLM 配置（支持多提供商）
├── embeddings.py            # 嵌入模块
├── loader.py                # 文档加载器
├── chunker.py               # 文本分块
├── vectorstore.py           # 向量存储
├── retriever.py             # 检索器
├── chain.py                 # RAG 链
│
├── # 工具系统
├── tools/
│   ├── base.py              # Tool 基类
│   ├── registry.py          # 工具注册中心
│   ├── knowledge_search.py  # 知识搜索工具
│   ├── examples.py          # 示例工具（Calculator, Echo等）
│   └── logger.py            # 工具日志
│
├── # Agent Runtime
├── agent_runtime.py         # 多工具调用运行时
│
├── # 规划 Agent
├── planning/
│   ├── task_decomposer.py   # 任务分解
│   ├── chain_of_thought.py  # 思维链推理
│   ├── self_reflection.py   # 自省反思
│   ├── planning_agent.py    # 规划 Agent
│   └── demo.py              # 演示脚本
│
├── # 记忆系统
├── memory/
│   ├── conversation.py      # 对话管理
│   ├── memory_system.py     # 记忆系统
│   └── conversational_agent.py  # 对话 Agent
│
├── # MCP Server
├── mcp/
│   ├── server.py            # MCP 服务器
│   ├── run.py               # 启动入口
│   └── test_server.py       # 测试脚本
│
└── tests/                   # 测试文件
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

### 方式二：命令行演示

```bash
python -m rag_minimal.main
```

### 方式三：Web 界面

```bash
streamlit run rag_minimal/app.py
```

### 方式四：MCP Server

```bash
python -m rag_minimal.mcp.run --docs-dir docs
```

## 使用示例

### 基础 RAG 查询

```python
from rag_minimal import AgentRuntime

runtime = AgentRuntime(docs_dir="docs")
result = runtime.ask("什么是 RAG？")
print(result.answer)
```

### 多工具并行调用

```python
from rag_minimal import AgentRuntime
from rag_minimal.tools import CalculatorTool, EchoTool
from rag_minimal.schemas import ToolCallRequest

runtime = AgentRuntime(docs_dir="docs")
runtime.register_tool(CalculatorTool())
runtime.register_tool(EchoTool())

# 并行调用多个工具
calls = [
    ToolCallRequest(tool_name="calculator", arguments={"expression": "2 + 3 * 4"}),
    ToolCallRequest(tool_name="echo", arguments={"message": "Hello", "uppercase": True}),
    ToolCallRequest(tool_name="knowledge_search", arguments={"query": "RAG", "top_k": 2}),
]
result = runtime.invoke_tools(calls, parallel=True)

print(f"成功: {result.successful_calls}/{result.total_calls}")
for r in result.results:
    print(f"  {r.tool_name}: {r.result}")
```

### 规划 Agent

```python
from rag_minimal.planning import PlanningAgent

agent = PlanningAgent(docs_dir="docs")
result = agent.run(
    query="分析 RAG 系统的优缺点并给出改进建议",
    use_decomposition=True,
    use_cot=True,
    use_reflection=True,
)
print(result.answer)
```

### 对话 Agent（带记忆）

```python
from rag_minimal.memory import ConversationalAgent

agent = ConversationalAgent(docs_dir="docs")

# 第一轮对话
result1 = agent.chat("什么是向量数据库？")
print(result1.response)

# 第二轮对话（自动使用历史上下文）
result2 = agent.chat("它有什么优点？", conversation_id=result1.conversation_id)
print(result2.response)
```

### 自定义工具

```python
from rag_minimal.tools import Tool
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    text: str = Field(..., description="输入文本")

class MyOutput(BaseModel):
    result: str = Field(..., description="处理结果")
    success: bool = True

class MyTool(Tool):
    name = "my_tool"
    description = "我的自定义工具"
    input_schema = MyInput
    output_schema = MyOutput
    
    def invoke(self, payload):
        validated = self.validate_input(payload)
        return MyOutput(result=f"处理: {validated.text}")

# 注册并使用
runtime.register_tool(MyTool())
result = runtime.invoke_tool("my_tool", {"text": "hello"})
```

## MCP 集成

### Claude Desktop 配置

在 Claude Desktop 配置文件中添加：

```json
{
  "mcpServers": {
    "rag-minimal": {
      "command": "python",
      "args": ["-m", "rag_minimal.mcp.run", "--docs-dir", "/path/to/docs"]
    }
  }
}
```

### 测试 MCP Server

```bash
python -m rag_minimal.mcp.test_server
```

## 运行测试

```bash
# 运行所有测试
pytest -q --cov=rag_minimal --cov-report=xml

# 运行特定模块测试
pytest rag_minimal/tests/test_planning.py -v
pytest rag_minimal/tests/test_memory.py -v
```

## 静态分析

```bash
ruff check rag_minimal/
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent Runtime                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   invoke_   │  │   invoke_   │  │    invoke_chain     │ │
│  │    tool     │  │    tools    │  │  (sequential +      │ │
│  │  (single)   │  │ (parallel)  │  │   result passing)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼───────────────────┼─────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tool Registry                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ knowledge  │ │ calculator │ │    echo    │ │   ...    │ │
│  │  _search   │ │            │ │            │ │          │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Export Formats                           │
│  ┌────────────────────┐  ┌────────────────────────────────┐│
│  │  OpenAI Function   │  │         MCP Protocol           ││
│  │     Calling        │  │    (JSON-RPC over stdio)       ││
│  └────────────────────┘  └────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| langchain | LLM 应用框架 |
| chromadb | 向量数据库 |
| pydantic | 数据验证和 JSON Schema |
| python-docx | Word 文档支持 |
| pypdf | PDF 文档支持 |
| streamlit | Web 界面 |
| pytest | 测试框架 |
| ruff | 代码检查 |

## 注意事项

- 默认使用 `SimpleLLM` 和 `FakeEmbeddings` 用于演示，无需 API 密钥
- 生产环境请替换为真实的 LLM 和 Embedding 模型
- MCP Server 使用纯 Python 实现，无需额外 SDK

## License

MIT
