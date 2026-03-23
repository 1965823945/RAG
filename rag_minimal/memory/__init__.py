"""Memory module for conversational AI with multi-turn dialogue support.

This module provides:
1. ConversationManager - Manages conversation sessions and message history
2. MemorySystem - Short-term, long-term, and working memory
3. ConversationalAgent - Agent with memory and context awareness
"""

from rag_minimal.memory.conversation import ConversationManager
from rag_minimal.memory.conversational_agent import ConversationalAgent
from rag_minimal.memory.memory_system import (
    LongTermMemory,
    MemorySystem,
    ShortTermMemory,
    WorkingMemory,
)

__all__ = [
    "ConversationManager",
    "MemorySystem",
    "ShortTermMemory",
    "LongTermMemory",
    "WorkingMemory",
    "ConversationalAgent",
]
