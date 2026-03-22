#!/usr/bin/env python
"""Test script for MCP Server.

This script simulates an MCP client to test the server implementation.
"""

import json
import sys
import os

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_minimal.mcp.server import MCPServer, JsonRpcRequest  # noqa: E402
from rag_minimal.tools.registry import ToolRegistry  # noqa: E402
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool  # noqa: E402


def test_mcp_server():
    """Test MCP server functionality."""
    print("=" * 60)
    print("MCP Server Test")
    print("=" * 60)

    # Create registry and server
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(docs_dir="docs"))
    server = MCPServer(registry=registry)

    # Test 1: Initialize
    print("\n[Test 1] Initialize")
    req = JsonRpcRequest(
        method="initialize",
        params={
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
        id=1,
    )
    resp = server.handle_request(req)
    print(f"  Response: {json.dumps(resp.to_dict(), indent=2)}")
    assert resp.result is not None
    assert "protocolVersion" in resp.result
    print("  [OK] Initialize OK")

    # Test 2: tools/list
    print("\n[Test 2] tools/list")
    req = JsonRpcRequest(method="tools/list", params={}, id=2)
    resp = server.handle_request(req)
    print(f"  Response: {json.dumps(resp.to_dict(), indent=2)}")
    assert resp.result is not None
    assert "tools" in resp.result
    assert len(resp.result["tools"]) > 0
    print(f"  [OK] Listed {len(resp.result['tools'])} tools")

    # Test 3: tools/call - knowledge_search
    print("\n[Test 3] tools/call - knowledge_search")
    req = JsonRpcRequest(
        method="tools/call",
        params={
            "name": "knowledge_search",
            "arguments": {"query": "RAG", "top_k": 2},
        },
        id=3,
    )
    resp = server.handle_request(req)
    print(f"  Response ID: {resp.id}")
    print(f"  Has content: {'content' in resp.result}")
    print(f"  Is error: {resp.result.get('isError', False)}")

    # Parse the result content
    if resp.result and "content" in resp.result:
        content = resp.result["content"][0]["text"]
        result_data = json.loads(content)
        print(f"  Success: {result_data.get('success')}")
        print(f"  Results count: {len(result_data.get('results', []))}")
        if result_data.get("results"):
            print(
                f"  First result chunk_id: {result_data['results'][0].get('chunk_id')}"
            )
    print("  [OK] tools/call OK")

    # Test 4: tools/call - non-existent tool
    print("\n[Test 4] tools/call - non-existent tool")
    req = JsonRpcRequest(
        method="tools/call",
        params={
            "name": "non_existent_tool",
            "arguments": {},
        },
        id=4,
    )
    resp = server.handle_request(req)
    assert resp.result.get("isError") is True
    print("  [OK] Error handling OK")

    # Test 5: Method not found
    print("\n[Test 5] Unknown method")
    req = JsonRpcRequest(method="unknown/method", params={}, id=5)
    resp = server.handle_request(req)
    assert resp.error is not None
    print(f"  Error code: {resp.error.get('code')}")
    print("  [OK] Method not found handled OK")

    # Test 6: Ping
    print("\n[Test 6] Ping")
    req = JsonRpcRequest(method="ping", params={}, id=6)
    resp = server.handle_request(req)
    assert resp.result is not None
    print("  [OK] Ping OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Show MCP tool format
    print("\n[MCP Tool Export Format]")
    for tool in registry:
        mcp_format = tool.to_mcp_tool()
        print(json.dumps(mcp_format, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_mcp_server()
