#!/usr/bin/env python
"""MCP Server entry point.

Usage:
    python -m rag_minimal.mcp.run [--docs-dir DOCS_DIR]

This starts the MCP server using stdio transport.
Tools registered in the ToolRegistry will be exposed via MCP protocol.
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_minimal.mcp.server import MCPServer
from rag_minimal.tools.registry import ToolRegistry
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool


def create_registry(docs_dir: str = "docs") -> ToolRegistry:
    """Create and populate the tool registry.

    Args:
        docs_dir: Directory containing documents for knowledge search

    Returns:
        ToolRegistry with registered tools
    """
    registry = ToolRegistry()

    # Register knowledge search tool
    registry.register(KnowledgeSearchTool(docs_dir=docs_dir))

    # Add more tools here as needed
    # registry.register(WebSearchTool())
    # registry.register(CodeExecutionTool())

    return registry


def main():
    parser = argparse.ArgumentParser(description="RAG Minimal MCP Server")
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory containing documents to search (default: docs)",
    )
    args = parser.parse_args()

    # Create registry with tools
    registry = create_registry(docs_dir=args.docs_dir)

    # Create and run MCP server
    server = MCPServer(registry=registry)
    server.run_stdio()


if __name__ == "__main__":
    main()
