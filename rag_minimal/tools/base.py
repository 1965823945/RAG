"""Base tool interface with JSON Schema support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, List, Optional
from pydantic import BaseModel

from rag_minimal.schemas import ToolMetadata


class Tool(ABC):
    """Standard tool interface.

    Every tool must define:
    - name: Unique identifier for the tool
    - description: Human-readable description
    - input_schema: Pydantic model for input validation
    - output_schema: Pydantic model for output

    And implement:
    - invoke(payload) -> output
    """

    name: str
    description: str
    version: str = "1.0.0"
    tags: Optional[List[str]] = None  # Avoid mutable default argument
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    def invoke(self, payload: Dict[str, Any]) -> BaseModel:
        """Execute the tool with validated input.

        Args:
            payload: Dictionary matching input_schema

        Returns:
            Output matching output_schema
        """
        raise NotImplementedError

    def get_input_json_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for tool input.

        Returns:
            JSON Schema dict compatible with OpenAPI/MCP
        """
        return self.input_schema.model_json_schema()

    def get_output_json_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for tool output.

        Returns:
            JSON Schema dict
        """
        return self.output_schema.model_json_schema()

    def get_metadata(self) -> ToolMetadata:
        """Get full tool metadata.

        Returns:
            ToolMetadata with name, description, schemas, etc.
        """
        return ToolMetadata(
            name=self.name,
            description=self.description,
            version=getattr(self, "version", "1.0.0"),
            input_schema=self.get_input_json_schema(),
            output_schema=self.get_output_json_schema(),
            tags=getattr(self, "tags", None) or [],
        )

    def to_openai_function(self) -> Dict[str, Any]:
        """Export as OpenAI function calling format.

        Returns:
            Dict compatible with OpenAI's function calling API
        """
        input_schema = self.get_input_json_schema()

        # Remove pydantic-specific fields
        if "$defs" in input_schema:
            del input_schema["$defs"]
        if "title" in input_schema:
            del input_schema["title"]

        return {
            "name": self.name,
            "description": self.description,
            "parameters": input_schema,
        }

    def to_mcp_tool(self) -> Dict[str, Any]:
        """Export as MCP tool format.

        Returns:
            Dict compatible with Model Context Protocol
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.get_input_json_schema(),
        }

    def validate_input(self, payload: Dict[str, Any]) -> BaseModel:
        """Validate input against schema.

        Args:
            payload: Raw input dict

        Returns:
            Validated pydantic model

        Raises:
            ValidationError if invalid
        """
        return self.input_schema(**payload)

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
