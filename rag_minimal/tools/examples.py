"""Example tools for demonstration and testing.

These tools serve as examples and can be used to test multi-tool invocation.
"""

from typing import Any

from pydantic import BaseModel, Field

from rag_minimal.schemas import ToolOutput
from rag_minimal.tools.base import Tool

# ─────────────────────────────────────────────────────────────
# Calculator Tool
# ─────────────────────────────────────────────────────────────


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(..., description="Mathematical expression to evaluate")

    class Config:
        extra = "forbid"


class CalculatorOutput(ToolOutput):
    """Output from calculator tool."""

    expression: str = Field(default="", description="Original expression")
    result: float = Field(default=0.0, description="Calculation result")


class CalculatorTool(Tool):
    """A simple calculator tool for mathematical expressions.

    Supports basic operations: +, -, *, /, **, (), and common math functions.

    Example:
        tool = CalculatorTool()
        result = tool.invoke({"expression": "2 + 3 * 4"})
        # result.result == 14.0
    """

    name = "calculator"
    description = (
        "Evaluate mathematical expressions. Supports +, -, *, /, **, parentheses."
    )
    version = "1.0.0"
    tags = ["math", "utility"]
    input_schema = CalculatorInput
    output_schema = CalculatorOutput

    # Allowed names for eval safety
    SAFE_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }

    def invoke(self, payload: dict[str, Any]) -> CalculatorOutput:
        """Evaluate the mathematical expression."""
        validated = self.validate_input(payload)
        expression = validated.expression

        try:
            # Safe eval with limited builtins
            result = eval(expression, {"__builtins__": {}}, self.SAFE_NAMES)
            return CalculatorOutput(
                success=True,
                message="ok",
                expression=expression,
                result=float(result),
            )
        except Exception as e:
            return CalculatorOutput(
                success=False,
                message=f"Calculation error: {e}",
                expression=expression,
                result=0.0,
            )


# ─────────────────────────────────────────────────────────────
# Echo Tool
# ─────────────────────────────────────────────────────────────


class EchoInput(BaseModel):
    """Input for echo tool."""

    message: str = Field(..., description="Message to echo")
    uppercase: bool = Field(default=False, description="Convert to uppercase")
    repeat: int = Field(default=1, ge=1, le=10, description="Number of times to repeat")

    class Config:
        extra = "forbid"


class EchoOutput(ToolOutput):
    """Output from echo tool."""

    original: str = Field(default="", description="Original message")
    echoed: str = Field(default="", description="Echoed message")


class EchoTool(Tool):
    """A simple echo tool for testing.

    Returns the input message, optionally transformed.

    Example:
        tool = EchoTool()
        result = tool.invoke({"message": "hello", "uppercase": True})
        # result.echoed == "HELLO"
    """

    name = "echo"
    description = "Echo back the input message with optional transformations."
    version = "1.0.0"
    tags = ["utility", "test"]
    input_schema = EchoInput
    output_schema = EchoOutput

    def invoke(self, payload: dict[str, Any]) -> EchoOutput:
        """Echo the message."""
        validated = self.validate_input(payload)

        message = validated.message
        if validated.uppercase:
            message = message.upper()
        if validated.repeat > 1:
            message = " ".join([message] * validated.repeat)

        return EchoOutput(
            success=True,
            message="ok",
            original=validated.message,
            echoed=message,
        )


# ─────────────────────────────────────────────────────────────
# Text Transform Tool
# ─────────────────────────────────────────────────────────────


class TextTransformInput(BaseModel):
    """Input for text transform tool."""

    text: str = Field(..., description="Text to transform")
    operation: str = Field(
        default="none",
        description="Operation: none, upper, lower, title, reverse, length",
    )

    class Config:
        extra = "forbid"


class TextTransformOutput(ToolOutput):
    """Output from text transform tool."""

    original: str = Field(default="", description="Original text")
    transformed: str = Field(default="", description="Transformed text")
    operation: str = Field(default="", description="Operation applied")


class TextTransformTool(Tool):
    """Transform text with various operations.

    Supported operations:
    - none: No transformation
    - upper: Convert to uppercase
    - lower: Convert to lowercase
    - title: Convert to title case
    - reverse: Reverse the text
    - length: Return the length as string
    """

    name = "text_transform"
    description = "Transform text: upper, lower, title, reverse, or get length."
    version = "1.0.0"
    tags = ["text", "utility"]
    input_schema = TextTransformInput
    output_schema = TextTransformOutput

    OPERATIONS = {
        "none": lambda x: x,
        "upper": lambda x: x.upper(),
        "lower": lambda x: x.lower(),
        "title": lambda x: x.title(),
        "reverse": lambda x: x[::-1],
        "length": lambda x: str(len(x)),
    }

    def invoke(self, payload: dict[str, Any]) -> TextTransformOutput:
        """Transform the text."""
        validated = self.validate_input(payload)

        operation = validated.operation.lower()
        if operation not in self.OPERATIONS:
            return TextTransformOutput(
                success=False,
                message=f"Unknown operation: {operation}. Supported: {list(self.OPERATIONS.keys())}",
                original=validated.text,
                operation=operation,
            )

        transformed = self.OPERATIONS[operation](validated.text)

        return TextTransformOutput(
            success=True,
            message="ok",
            original=validated.text,
            transformed=transformed,
            operation=operation,
        )


# ─────────────────────────────────────────────────────────────
# List Aggregator Tool (for chaining demos)
# ─────────────────────────────────────────────────────────────


class ListAggregatorInput(BaseModel):
    """Input for list aggregator tool."""

    items: list[str] = Field(default_factory=list, description="Items to aggregate")
    separator: str = Field(default=", ", description="Separator between items")
    result: dict[str, Any] = Field(
        default_factory=dict, description="Previous result (for chaining)"
    )

    class Config:
        extra = "forbid"


class ListAggregatorOutput(ToolOutput):
    """Output from list aggregator tool."""

    items: list[str] = Field(default_factory=list, description="Input items")
    aggregated: str = Field(default="", description="Aggregated string")
    count: int = Field(default=0, description="Number of items")


class ListAggregatorTool(Tool):
    """Aggregate a list of items into a single string.

    Useful for chaining multiple tool outputs together.
    """

    name = "list_aggregator"
    description = "Aggregate a list of items into a single string."
    version = "1.0.0"
    tags = ["utility", "aggregation"]
    input_schema = ListAggregatorInput
    output_schema = ListAggregatorOutput

    def invoke(self, payload: dict[str, Any]) -> ListAggregatorOutput:
        """Aggregate the items."""
        validated = self.validate_input(payload)

        # If previous result is passed, try to extract items from it
        items = list(validated.items)
        if validated.result:
            # Try to extract echoed/transformed from previous result
            if "echoed" in validated.result:
                items.append(validated.result["echoed"])
            elif "transformed" in validated.result:
                items.append(validated.result["transformed"])
            elif "result" in validated.result:
                items.append(str(validated.result["result"]))

        aggregated = validated.separator.join(items)

        return ListAggregatorOutput(
            success=True,
            message="ok",
            items=items,
            aggregated=aggregated,
            count=len(items),
        )


# Export all tools
__all__ = [
    "CalculatorTool",
    "CalculatorInput",
    "CalculatorOutput",
    "EchoTool",
    "EchoInput",
    "EchoOutput",
    "TextTransformTool",
    "TextTransformInput",
    "TextTransformOutput",
    "ListAggregatorTool",
    "ListAggregatorInput",
    "ListAggregatorOutput",
]
