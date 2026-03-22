"""Shared schemas for standardized tool interfaces."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base tool input schema."""

    pass


class ToolOutput(BaseModel):
    """Base tool output schema."""

    success: bool = Field(default=True)
    message: str = Field(default="ok")


class SearchInput(BaseModel):
    """Input for knowledge search tool."""

    query: str = Field(..., description="User question or query")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of results to return"
    )


class SearchResultItem(BaseModel):
    """Single search result."""

    content: str
    source: str = ""
    score: float = 0.0


class SearchOutput(BaseModel):
    """Output for knowledge search tool."""

    success: bool = True
    message: str = "ok"
    query: str = ""
    results: List[SearchResultItem] = Field(default_factory=list)
