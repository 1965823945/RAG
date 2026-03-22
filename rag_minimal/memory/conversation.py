"""Conversation Manager - Manages conversation sessions and message history."""

import uuid
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from rag_minimal.schemas import (
    Message,
    MessageRole,
    Conversation,
    SearchResultItem,
)


class ConversationManager:
    """Manages conversation sessions and message history.

    Features:
    1. Create and manage multiple conversation sessions
    2. Add messages to conversations
    3. Get conversation history with configurable window
    4. Persist conversations to disk (optional)
    5. Generate conversation summaries
    """

    def __init__(
        self,
        max_history_messages: int = 20,
        persist_dir: Optional[str] = None,
    ):
        """Initialize the conversation manager.

        Args:
            max_history_messages: Maximum messages to keep in memory per conversation
            persist_dir: Directory to persist conversations (None for in-memory only)
        """
        self.max_history_messages = max_history_messages
        self.persist_dir = Path(persist_dir) if persist_dir else None

        # In-memory storage
        self._conversations: Dict[str, Conversation] = {}

        # Load persisted conversations
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_conversations()

    def create_conversation(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation session.

        Args:
            title: Optional title for the conversation
            metadata: Optional metadata

        Returns:
            New Conversation object
        """
        conversation_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        conversation = Conversation(
            id=conversation_id,
            title=title or f"对话 {conversation_id[:6]}",
            messages=[],
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            message_count=0,
            token_count=0,
        )

        self._conversations[conversation_id] = conversation
        self._persist_conversation(conversation)

        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            Conversation if found, None otherwise
        """
        return self._conversations.get(conversation_id)

    def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Get an existing conversation or create a new one.

        Args:
            conversation_id: Optional conversation ID to retrieve

        Returns:
            Existing or new Conversation
        """
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        return self.create_conversation()

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        sources: Optional[List[SearchResultItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """Add a message to a conversation.

        Args:
            conversation_id: The conversation ID
            role: Message role (user, assistant, system, tool)
            content: Message content
            sources: Optional sources for assistant messages
            metadata: Optional metadata
            tool_name: Tool name for tool messages
            tool_call_id: Tool call ID for tool messages

        Returns:
            The created Message

        Raises:
            ValueError: If conversation not found
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        message_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        message = Message(
            id=message_id,
            role=role,
            content=content,
            timestamp=now,
            metadata=metadata or {},
            sources=sources or [],
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

        conversation.messages.append(message)
        conversation.message_count = len(conversation.messages)
        conversation.updated_at = now

        # Estimate token count (rough approximation)
        conversation.token_count += self._estimate_tokens(content)

        # Trim old messages if exceeding limit
        if len(conversation.messages) > self.max_history_messages:
            removed = conversation.messages.pop(0)
            conversation.token_count -= self._estimate_tokens(removed.content)

        self._persist_conversation(conversation)

        return message

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Convenience method to add a user message."""
        return self.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: Optional[List[SearchResultItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Convenience method to add an assistant message."""
        return self.add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            sources=sources,
            metadata=metadata,
        )

    def add_system_message(
        self,
        conversation_id: str,
        content: str,
    ) -> Message:
        """Convenience method to add a system message."""
        return self.add_message(
            conversation_id=conversation_id,
            role=MessageRole.SYSTEM,
            content=content,
        )

    def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        include_system: bool = False,
    ) -> List[Message]:
        """Get conversation history.

        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages to return (from most recent)
            include_system: Whether to include system messages

        Returns:
            List of messages
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        messages = conversation.messages

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_history_as_text(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        format_style: str = "chat",
    ) -> str:
        """Get conversation history as formatted text.

        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages
            format_style: 'chat' or 'qa'

        Returns:
            Formatted history string
        """
        messages = self.get_history(conversation_id, limit)

        if not messages:
            return ""

        lines = []
        for msg in messages:
            if format_style == "chat":
                role_name = {
                    MessageRole.USER: "用户",
                    MessageRole.ASSISTANT: "助手",
                    MessageRole.SYSTEM: "系统",
                    MessageRole.TOOL: "工具",
                }.get(msg.role, "未知")
                lines.append(f"{role_name}: {msg.content}")
            elif format_style == "qa":
                if msg.role == MessageRole.USER:
                    lines.append(f"Q: {msg.content}")
                elif msg.role == MessageRole.ASSISTANT:
                    lines.append(f"A: {msg.content}")

        return "\n".join(lines)

    def get_last_user_message(self, conversation_id: str) -> Optional[Message]:
        """Get the last user message in a conversation."""
        messages = self.get_history(conversation_id)
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                return msg
        return None

    def get_last_assistant_message(self, conversation_id: str) -> Optional[Message]:
        """Get the last assistant message in a conversation."""
        messages = self.get_history(conversation_id)
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None

    def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """Update the conversation title."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.title = title
            conversation.updated_at = datetime.now().isoformat()
            self._persist_conversation(conversation)

    def update_conversation_summary(
        self,
        conversation_id: str,
        summary: str,
    ) -> None:
        """Update the conversation summary."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.summary = summary
            conversation.updated_at = datetime.now().isoformat()
            self._persist_conversation(conversation)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Returns:
            True if deleted, False if not found
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

            if self.persist_dir:
                file_path = self.persist_dir / f"{conversation_id}.json"
                if file_path.exists():
                    file_path.unlink()

            return True
        return False

    def list_conversations(
        self,
        limit: int = 50,
        sort_by: str = "updated_at",
    ) -> List[Conversation]:
        """List all conversations.

        Args:
            limit: Maximum conversations to return
            sort_by: Sort field ('updated_at' or 'created_at')

        Returns:
            List of conversations
        """
        conversations = list(self._conversations.values())

        if sort_by == "updated_at":
            conversations.sort(key=lambda c: c.updated_at, reverse=True)
        elif sort_by == "created_at":
            conversations.sort(key=lambda c: c.created_at, reverse=True)

        return conversations[:limit]

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear all messages from a conversation but keep it.

        Returns:
            True if cleared, False if not found
        """
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.messages = []
            conversation.message_count = 0
            conversation.token_count = 0
            conversation.updated_at = datetime.now().isoformat()
            self._persist_conversation(conversation)
            return True
        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        For Chinese: ~1.5 chars per token
        For English: ~4 chars per token
        """
        # Count Chinese characters
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars

        return int(chinese_chars / 1.5) + int(other_chars / 4)

    def _persist_conversation(self, conversation: Conversation) -> None:
        """Persist a conversation to disk."""
        if not self.persist_dir:
            return

        file_path = self.persist_dir / f"{conversation.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_conversations(self) -> None:
        """Load conversations from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return

        for file_path in self.persist_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversation = Conversation(**data)
                    self._conversations[conversation.id] = conversation
            except Exception as e:
                print(f"Failed to load conversation from {file_path}: {e}")

    def generate_title_from_first_message(
        self,
        conversation_id: str,
        max_length: int = 30,
    ) -> Optional[str]:
        """Generate a title from the first user message.

        Args:
            conversation_id: The conversation ID
            max_length: Maximum title length

        Returns:
            Generated title or None
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None

        # Find first user message
        for msg in conversation.messages:
            if msg.role == MessageRole.USER:
                title = msg.content[:max_length]
                if len(msg.content) > max_length:
                    title = title.rsplit(" ", 1)[0] + "..."
                self.update_conversation_title(conversation_id, title)
                return title

        return None
