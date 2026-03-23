"""MCP (Model Context Protocol) Server implementation.

This is a lightweight implementation of the MCP protocol using JSON-RPC over stdio.
No external MCP SDK required.

Protocol spec: https://modelcontextprotocol.io/
"""

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rag_minimal.tools.registry import ToolRegistry

# Configure logging to stderr (stdout is for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# JSON-RPC Types
# ─────────────────────────────────────────────────────────────


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 Request."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | str | None = None
    jsonrpc: str = "2.0"


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 Response."""

    id: int | str | None
    result: Any | None = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


# JSON-RPC Error Codes
class JsonRpcError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ─────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────


class MCPServer:
    """MCP Server that exposes tools via JSON-RPC over stdio.

    Implements the Model Context Protocol for tool discovery and invocation.
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "rag-minimal-mcp"
    SERVER_VERSION = "1.0.0"

    def __init__(self, registry: ToolRegistry):
        """Initialize MCP Server.

        Args:
            registry: ToolRegistry containing tools to expose
        """
        self.registry = registry
        self._handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
        }

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        logger.info(f"Client initializing: {params.get('clientInfo', {})}")
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},  # We support tools
            },
            "serverInfo": {
                "name": self.SERVER_NAME,
                "version": self.SERVER_VERSION,
            },
        }

    def _handle_initialized(self, params: dict[str, Any]) -> None:
        """Handle initialized notification."""
        logger.info("Client initialized successfully")
        return None

    def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ping request."""
        return {}

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request.

        Returns list of available tools in MCP format.
        """
        tools = []
        for tool in self.registry:
            tools.append(tool.to_mcp_tool())

        logger.info(f"Listed {len(tools)} tools")
        return {"tools": tools}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request.

        Invokes the specified tool and returns the result.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"Calling tool: {tool_name} with args: {list(arguments.keys())}")

        # Get tool
        tool = self.registry.get_optional(tool_name)
        if tool is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{tool_name}' not found",
                    }
                ],
                "isError": True,
            }

        # Invoke tool
        try:
            result = tool.invoke(arguments)

            # Convert result to MCP content format
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = (
                    dict(result)
                    if hasattr(result, "__dict__")
                    else {"result": str(result)}
                )

            # Check if tool returned an error
            is_error = not result_dict.get("success", True)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result_dict, ensure_ascii=False, indent=2),
                    }
                ],
                "isError": is_error,
            }

        except Exception as e:
            logger.error(f"Tool invocation failed: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error invoking tool: {str(e)}",
                    }
                ],
                "isError": True,
            }

    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse | None:
        """Handle a single JSON-RPC request.

        Args:
            request: Parsed JSON-RPC request

        Returns:
            JsonRpcResponse or None for notifications
        """
        handler = self._handlers.get(request.method)

        if handler is None:
            if request.id is not None:
                return JsonRpcResponse(
                    id=request.id,
                    error={
                        "code": JsonRpcError.METHOD_NOT_FOUND,
                        "message": f"Method not found: {request.method}",
                    },
                )
            return None

        try:
            result = handler(request.params)

            # Notifications don't get responses
            if request.id is None:
                return None

            return JsonRpcResponse(id=request.id, result=result)

        except Exception as e:
            logger.error(f"Handler error: {e}")
            if request.id is not None:
                return JsonRpcResponse(
                    id=request.id,
                    error={
                        "code": JsonRpcError.INTERNAL_ERROR,
                        "message": str(e),
                    },
                )
            return None

    def run_stdio(self):
        """Run the MCP server using stdio transport.

        Reads JSON-RPC requests from stdin, writes responses to stdout.
        """
        logger.info(f"MCP Server starting ({self.SERVER_NAME} v{self.SERVER_VERSION})")
        logger.info(f"Available tools: {self.registry.list_tools()}")

        while True:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    logger.info("EOF received, shutting down")
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON-RPC request
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    response = JsonRpcResponse(
                        id=None,
                        error={
                            "code": JsonRpcError.PARSE_ERROR,
                            "message": f"Parse error: {e}",
                        },
                    )
                    self._send_response(response)
                    continue

                # Create request object
                request = JsonRpcRequest(
                    method=data.get("method", ""),
                    params=data.get("params", {}),
                    id=data.get("id"),
                    jsonrpc=data.get("jsonrpc", "2.0"),
                )

                # Handle request
                response = self.handle_request(request)

                # Send response if not a notification
                if response is not None:
                    self._send_response(response)

            except KeyboardInterrupt:
                logger.info("Interrupted, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

    def _send_response(self, response: JsonRpcResponse):
        """Send a JSON-RPC response to stdout."""
        output = json.dumps(response.to_dict(), ensure_ascii=False)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
