"""Conversational Agent - Agent with memory and multi-turn dialogue support."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseLLM

from rag_minimal.memory.conversation import ConversationManager
from rag_minimal.memory.memory_system import MemorySystem
from rag_minimal.schemas import (
    Conversation,
    ConversationalOutput,
    MemoryType,
    MessageRole,
    SearchOutput,
    SearchResultItem,
)

# Prompt templates
CONVERSATIONAL_SYSTEM_PROMPT = """你是一个智能助手，能够进行多轮对话并记住之前的上下文。

请注意：
1. 参考对话历史来理解上下文
2. 利用已知的用户偏好和记忆
3. 基于检索到的文档提供准确的回答
4. 如果不确定，请诚实地表示

当前时间: {current_time}"""

CONVERSATIONAL_RAG_PROMPT = """基于以下信息回答用户的问题。

{context}

用户当前问题: {question}

请提供有帮助的回答:"""

SUMMARY_PROMPT = """请为以下对话生成一个简短的摘要（50字以内）：

{conversation}

摘要:"""


class ConversationalAgent:
    """Agent with memory and multi-turn dialogue support.

    Features:
    1. Multi-turn conversation with history tracking
    2. Short-term and long-term memory
    3. Context-aware responses
    4. Automatic memory extraction
    5. Conversation summarization
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        search_func: Callable[[str, int], SearchOutput] | None = None,
        persist_dir: str | None = None,
        max_history_messages: int = 20,
        max_context_tokens: int = 4000,
        auto_extract_memories: bool = True,
    ):
        """Initialize the conversational agent.

        Args:
            llm: Language model for generation
            search_func: Function to search knowledge base
            persist_dir: Directory for persistence
            max_history_messages: Maximum messages to keep per conversation
            max_context_tokens: Maximum tokens for context
            auto_extract_memories: Whether to auto-extract memories from messages
        """
        self.llm = llm
        self.search_func = search_func
        self.auto_extract_memories = auto_extract_memories

        # Initialize conversation manager
        self.conversations = ConversationManager(
            max_history_messages=max_history_messages,
            persist_dir=f"{persist_dir}/conversations" if persist_dir else None,
        )

        # Initialize memory system
        self.memory = MemorySystem(
            persist_dir=f"{persist_dir}/memory" if persist_dir else None,
            max_context_tokens=max_context_tokens,
        )

        # Configuration
        self.max_context_tokens = max_context_tokens

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        top_k: int = 3,
        include_history: bool = True,
        include_memories: bool = True,
    ) -> ConversationalOutput:
        """Process a chat message and generate a response.

        Args:
            message: User message
            conversation_id: Conversation ID (creates new if None)
            top_k: Number of documents to retrieve
            include_history: Whether to include conversation history
            include_memories: Whether to include relevant memories

        Returns:
            ConversationalOutput with response and metadata
        """
        # Get or create conversation
        conversation = self.conversations.get_or_create_conversation(conversation_id)
        conversation_id = conversation.id

        # Add user message
        user_msg = self.conversations.add_user_message(
            conversation_id=conversation_id,
            content=message,
        )

        # Extract and store memories from user message
        new_memories = []
        if self.auto_extract_memories:
            extracted = self.memory.extract_and_store_memories(
                message=message,
                is_user=True,
                conversation_id=conversation_id,
                message_id=user_msg.id,
            )
            new_memories = [e.content for e in extracted]

        # Build context
        context_parts = []
        messages_used = 0
        memories_used = 0
        documents_used = 0
        sources: list[SearchResultItem] = []

        # 1. Add conversation history
        if include_history:
            history = self.conversations.get_history(
                conversation_id=conversation_id,
                limit=6,  # Recent messages
            )
            # Exclude the current message
            history = [m for m in history if m.id != user_msg.id]
            if history:
                history_text = self.conversations.get_history_as_text(
                    conversation_id=conversation_id,
                    limit=6,
                )
                if history_text:
                    context_parts.append(f"对话历史:\n{history_text}")
                    messages_used = len(history)

        # 2. Add relevant memories
        if include_memories:
            memory_results = self.memory.recall(
                query=message,
                limit=5,
                include_short_term=True,
                include_long_term=True,
            )
            if memory_results:
                memory_texts = []
                for r in memory_results:
                    memory_texts.append(
                        f"- [{r.entry.memory_type.value}] {r.entry.content}"
                    )
                context_parts.append("相关记忆:\n" + "\n".join(memory_texts))
                memories_used = len(memory_results)

        # 3. Search knowledge base
        if self.search_func:
            # Enhance query with context
            enhanced_query = self._enhance_query(message, conversation_id)

            search_result = self.search_func(enhanced_query, top_k)
            if search_result.success and search_result.results:
                doc_texts = []
                for item in search_result.results:
                    doc_texts.append(f"[{item.source}] {item.content}")
                    sources.append(item)
                context_parts.append("参考文档:\n" + "\n\n".join(doc_texts))
                documents_used = len(search_result.results)

        # Build full context
        context = "\n\n".join(context_parts) if context_parts else "无额外上下文"

        # Generate response
        response = self._generate_response(message, context)

        # Add assistant message
        assistant_msg = self.conversations.add_assistant_message(
            conversation_id=conversation_id,
            content=response,
            sources=sources,
        )

        # Extract memories from assistant response
        if self.auto_extract_memories:
            self.memory.extract_and_store_memories(
                message=response,
                is_user=False,
                conversation_id=conversation_id,
                message_id=assistant_msg.id,
            )

        # Generate title if first exchange
        if conversation.message_count <= 2:
            self.conversations.generate_title_from_first_message(conversation_id)

        # Determine entities mentioned
        entities_mentioned = self._extract_entities_mentioned(message + " " + response)

        return ConversationalOutput(
            success=True,
            message="ok",
            conversation_id=conversation_id,
            message_id=assistant_msg.id,
            response=response,
            messages_used=messages_used,
            memories_used=memories_used,
            documents_used=documents_used,
            sources=sources,
            new_memories=new_memories,
            entities_mentioned=entities_mentioned,
        )

    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using LLM or fallback."""
        if not self.llm:
            return self._simple_response(question, context)

        # Build prompt
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

        user_prompt = CONVERSATIONAL_RAG_PROMPT.format(
            context=context,
            question=question,
        )

        # Generate with LLM
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            result = self.llm.invoke(full_prompt)

            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def _simple_response(self, question: str, context: str) -> str:
        """Generate simple response without LLM."""
        if "参考文档" in context:
            # Extract document content
            import re

            doc_match = re.search(r"参考文档:\n(.+?)(?:\n\n|$)", context, re.DOTALL)
            if doc_match:
                doc_content = doc_match.group(1)
                return f"根据检索到的信息：{doc_content[:300]}..."

        if "对话历史" in context:
            return "基于我们之前的对话，我理解您的问题。但我需要更多具体信息来提供准确的回答。"

        return f"您问的是关于「{question[:30]}」的问题。我需要更多上下文信息来回答。"

    def _enhance_query(
        self,
        query: str,
        conversation_id: str,
    ) -> str:
        """Enhance query with conversation context."""
        # Get recent context for pronouns resolution
        last_user = self.conversations.get_last_user_message(conversation_id)
        last_assistant = self.conversations.get_last_assistant_message(conversation_id)

        # Check if query has pronouns that need resolution
        pronouns = ["它", "这个", "那个", "他", "她", "这", "那", "上面", "之前"]
        has_pronoun = any(p in query for p in pronouns)

        if has_pronoun and (last_user or last_assistant):
            # Add context from previous messages
            context_terms = []
            if last_user:
                # Extract key terms from last user message
                import re

                terms = re.findall(r"[\u4e00-\u9fff]{2,}", last_user.content)
                context_terms.extend(terms[:3])

            if context_terms:
                enhanced = f"{query} {' '.join(context_terms)}"
                return enhanced

        return query

    def _extract_entities_mentioned(self, text: str) -> dict[str, str]:
        """Extract entities mentioned in text."""
        entities = {}
        import re

        # Simple patterns for common entities
        patterns = [
            (r"(\S+?)(?:公司|企业|集团)", "organization"),
            (r"(\S+?)(?:技术|系统|框架|语言)", "technology"),
            (r"(\S+?)(?:人|先生|女士|老师)", "person"),
        ]

        for pattern, entity_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches[:2]:  # Limit to 2 per type
                if len(match) > 1:
                    entities[match] = entity_type

        return entities

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self.conversations.get_conversation(conversation_id)

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        """List recent conversations."""
        return self.conversations.list_conversations(limit=limit)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self.conversations.delete_conversation(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's messages."""
        return self.conversations.clear_conversation(conversation_id)

    def get_memory_summary(self) -> dict[str, Any]:
        """Get memory system summary."""
        return self.memory.get_summary()

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.7,
        persist: bool = True,
    ) -> None:
        """Manually add a memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            persist: Whether to persist to long-term memory
        """
        self.memory.remember(
            content=content,
            memory_type=memory_type,
            importance=importance,
            persist=persist,
        )

    def search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of memory entries as dicts
        """
        results = self.memory.recall(query, limit=limit)
        return [
            {
                "content": r.entry.content,
                "type": r.entry.memory_type.value,
                "relevance": r.relevance_score,
                "importance": r.entry.importance,
            }
            for r in results
        ]

    def summarize_conversation(
        self,
        conversation_id: str,
    ) -> str | None:
        """Generate and store a conversation summary.

        Args:
            conversation_id: Conversation to summarize

        Returns:
            Generated summary or None
        """
        conversation = self.conversations.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None

        # Build conversation text
        history_text = self.conversations.get_history_as_text(
            conversation_id=conversation_id,
            limit=20,
        )

        if not history_text:
            return None

        # Generate summary
        if self.llm:
            try:
                prompt = SUMMARY_PROMPT.format(conversation=history_text)
                result = self.llm.invoke(prompt)

                if hasattr(result, "content"):
                    summary = result.content
                else:
                    summary = str(result)

                # Store summary
                self.conversations.update_conversation_summary(
                    conversation_id=conversation_id,
                    summary=summary,
                )

                # Also store in long-term memory
                self.memory.long_term.add_summary(
                    content=summary,
                    source_conversation_id=conversation_id,
                )

                return summary
            except Exception:
                pass

        # Simple summary fallback
        messages = conversation.messages
        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        if user_msgs:
            first_topic = user_msgs[0].content[:50]
            summary = f"关于「{first_topic}...」的对话，共{len(messages)}条消息"
            self.conversations.update_conversation_summary(conversation_id, summary)
            return summary

        return None

    def get_context_for_message(
        self,
        message: str,
        conversation_id: str | None = None,
        include_history: bool = True,
        include_memories: bool = True,
    ) -> str:
        """Get context string for a message without generating response.

        Useful for debugging or external processing.

        Args:
            message: The message to build context for
            conversation_id: Optional conversation ID
            include_history: Include conversation history
            include_memories: Include relevant memories

        Returns:
            Formatted context string
        """
        context_parts = []

        if include_history and conversation_id:
            history_text = self.conversations.get_history_as_text(
                conversation_id=conversation_id,
                limit=6,
            )
            if history_text:
                context_parts.append(f"对话历史:\n{history_text}")

        if include_memories:
            memories = self.memory.recall(message, limit=5)
            if memories:
                memory_texts = [
                    f"- [{r.entry.memory_type.value}] {r.entry.content}"
                    for r in memories
                ]
                context_parts.append("相关记忆:\n" + "\n".join(memory_texts))

        return "\n\n".join(context_parts) if context_parts else ""

    def export_conversation(
        self,
        conversation_id: str,
        format: str = "text",
    ) -> str | None:
        """Export a conversation.

        Args:
            conversation_id: Conversation to export
            format: Export format ('text', 'json', 'markdown')

        Returns:
            Exported content or None
        """
        conversation = self.conversations.get_conversation(conversation_id)
        if not conversation:
            return None

        if format == "text":
            return self.conversations.get_history_as_text(
                conversation_id=conversation_id,
                format_style="chat",
            )
        elif format == "json":
            import json

            return json.dumps(conversation.model_dump(), ensure_ascii=False, indent=2)
        elif format == "markdown":
            lines = [f"# {conversation.title or '对话记录'}", ""]
            for msg in conversation.messages:
                role = "**用户**" if msg.role == MessageRole.USER else "**助手**"
                lines.append(f"{role}: {msg.content}")
                lines.append("")
            return "\n".join(lines)

        return None
