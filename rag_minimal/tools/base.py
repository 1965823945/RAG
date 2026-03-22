"""Base tool interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel


class Tool(ABC):
    """Standard tool interface."""

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    def invoke(self, payload: Dict[str, Any]) -> BaseModel:
        """Execute the tool with validated input."""
        raise NotImplementedError
