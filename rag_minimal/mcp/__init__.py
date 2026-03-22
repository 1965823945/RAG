"""MCP package - Model Context Protocol implementation.

This package provides an MCP server that exposes RAG tools via JSON-RPC over stdio.

Usage:
    # Start MCP server
    python -m rag_minimal.mcp.run --docs-dir docs

    # Or programmatically
    from rag_minimal.mcp import MCPServer
    from rag_minimal.tools import ToolRegistry, KnowledgeSearchTool

    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(docs_dir="docs"))
    server = MCPServer(registry=registry)
    server.run_stdio()

For Claude Desktop / Cursor configuration, see mcp_config_example.json
"""

from rag_minimal.mcp.server import MCPServer, JsonRpcRequest, JsonRpcResponse

__all__ = ["MCPServer", "JsonRpcRequest", "JsonRpcResponse"]
