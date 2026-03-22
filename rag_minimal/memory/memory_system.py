"""Memory System - Short-term, Long-term, and Working Memory.

Provides a hierarchical memory system for the conversational agent:
1. ShortTermMemory - Current session context (volatile)
2. LongTermMemory - Persistent knowledge across sessions
3. WorkingMemory - Active context for current query
"""

import uuid
import json
import re
import hashlib
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from rag_minimal.schemas import (
    MemoryEntry,
    MemoryType,
    MemorySearchResult,
    Message,
    MessageRole,
)


class ShortTermMemory:
    """Short-term memory for current session.

    Stores recent context and information that should be available
    during the current conversation but may be forgotten later.
    """

    def __init__(
        self,
        capacity: int = 50,
        ttl_minutes: int = 60,
    ):
        """Initialize short-term memory.

        Args:
            capacity: Maximum number of entries
            ttl_minutes: Time-to-live in minutes for entries
        """
        self.capacity = capacity
        self.ttl_minutes = ttl_minutes
        self._entries: Dict[str, MemoryEntry] = {}
        self._access_order: List[str] = []  # LRU tracking

    def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        importance: float = 0.5,
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Add an entry to short-term memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            source_conversation_id: Source conversation
            source_message_id: Source message

        Returns:
            Created MemoryEntry
        """
        # Check capacity and evict if needed
        while len(self._entries) >= self.capacity:
            self._evict_oldest()

        entry_id = str(uuid.uuid4())[:12]
        now = datetime.now()
        expires_at = (now + timedelta(minutes=self.ttl_minutes)).isoformat()

        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            created_at=now.isoformat(),
            last_accessed=now.isoformat(),
            importance=importance,
            access_count=0,
            expires_at=expires_at,
        )

        self._entries[entry_id] = entry
        self._access_order.append(entry_id)

        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry and update access info."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.last_accessed = datetime.now().isoformat()
            entry.access_count += 1
            # Update LRU order
            if entry_id in self._access_order:
                self._access_order.remove(entry_id)
            self._access_order.append(entry_id)
        return entry

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemorySearchResult]:
        """Search short-term memory by keyword matching.

        Args:
            query: Search query
            limit: Maximum results
            memory_types: Filter by memory types

        Returns:
            List of search results with relevance scores
        """
        self._cleanup_expired()

        results = []
        query_lower = query.lower()
        query_keywords = set(self._extract_keywords(query))

        for entry in self._entries.values():
            if memory_types and entry.memory_type not in memory_types:
                continue

            # Calculate relevance score
            content_lower = entry.content.lower()
            content_keywords = set(self._extract_keywords(entry.content))

            # Keyword overlap
            overlap = len(query_keywords & content_keywords)
            keyword_score = overlap / max(len(query_keywords), 1)

            # Substring match
            substring_score = 1.0 if query_lower in content_lower else 0.0

            # Recency bonus
            try:
                created = datetime.fromisoformat(entry.created_at)
                age_minutes = (datetime.now() - created).total_seconds() / 60
                recency_score = max(0, 1 - age_minutes / self.ttl_minutes)
            except (ValueError, TypeError):
                recency_score = 0.5

            # Combined score
            relevance = (
                keyword_score * 0.4
                + substring_score * 0.3
                + recency_score * 0.2
                + entry.importance * 0.1
            )

            if relevance > 0.1:
                results.append(
                    MemorySearchResult(
                        entry=entry,
                        relevance_score=min(1.0, relevance),
                    )
                )

        # Sort by relevance and return top results
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent entries."""
        self._cleanup_expired()
        entries = list(self._entries.values())
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._entries.clear()
        self._access_order.clear()

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            oldest_id = self._access_order.pop(0)
            self._entries.pop(oldest_id, None)

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now()
        expired_ids = []

        for entry_id, entry in self._entries.items():
            if entry.expires_at:
                try:
                    expires = datetime.fromisoformat(entry.expires_at)
                    if now > expires:
                        expired_ids.append(entry_id)
                except (ValueError, TypeError):
                    pass

        for entry_id in expired_ids:
            self._entries.pop(entry_id, None)
            if entry_id in self._access_order:
                self._access_order.remove(entry_id)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Chinese and English words
        words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text.lower())
        return words


class LongTermMemory:
    """Long-term memory with persistence.

    Stores important information that should be retained across sessions.
    Supports entity extraction, fact storage, and semantic search.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
    ):
        """Initialize long-term memory.

        Args:
            persist_dir: Directory for persistence
            embedding_func: Function to generate embeddings for semantic search
        """
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_func = embedding_func

        self._entries: Dict[str, MemoryEntry] = {}
        self._entity_index: Dict[str, List[str]] = defaultdict(
            list
        )  # entity -> entry_ids
        self._type_index: Dict[MemoryType, List[str]] = defaultdict(
            list
        )  # type -> entry_ids

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Add an entry to long-term memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            entity_name: Entity name for entity memories
            entity_type: Entity type
            source_conversation_id: Source conversation
            source_message_id: Source message

        Returns:
            Created MemoryEntry
        """
        # Check for duplicate content
        content_hash = self._hash_content(content)
        for existing in self._entries.values():
            if self._hash_content(existing.content) == content_hash:
                # Update existing entry
                existing.last_accessed = datetime.now().isoformat()
                existing.access_count += 1
                if importance > existing.importance:
                    existing.importance = importance
                self._persist()
                return existing

        entry_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        # Generate embedding if function provided
        embedding = None
        if self.embedding_func:
            try:
                embedding = self.embedding_func(content)
            except Exception:
                pass

        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            created_at=now,
            last_accessed=now,
            importance=importance,
            access_count=0,
            entity_name=entity_name,
            entity_type=entity_type,
            embedding=embedding,
        )

        self._entries[entry_id] = entry
        self._type_index[memory_type].append(entry_id)

        if entity_name:
            self._entity_index[entity_name.lower()].append(entry_id)

        self._persist()
        return entry

    def add_fact(
        self,
        content: str,
        importance: float = 0.7,
        **kwargs,
    ) -> MemoryEntry:
        """Convenience method to add a fact."""
        return self.add(
            content=content,
            memory_type=MemoryType.FACT,
            importance=importance,
            **kwargs,
        )

    def add_preference(
        self,
        content: str,
        importance: float = 0.8,
        **kwargs,
    ) -> MemoryEntry:
        """Convenience method to add a user preference."""
        return self.add(
            content=content,
            memory_type=MemoryType.PREFERENCE,
            importance=importance,
            **kwargs,
        )

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        importance: float = 0.6,
        **kwargs,
    ) -> MemoryEntry:
        """Add entity information to memory."""
        return self.add(
            content=description,
            memory_type=MemoryType.ENTITY,
            importance=importance,
            entity_name=entity_name,
            entity_type=entity_type,
            **kwargs,
        )

    def add_summary(
        self,
        content: str,
        source_conversation_id: str,
        importance: float = 0.7,
    ) -> MemoryEntry:
        """Add a conversation summary."""
        return self.add(
            content=content,
            memory_type=MemoryType.SUMMARY,
            importance=importance,
            source_conversation_id=source_conversation_id,
        )

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.last_accessed = datetime.now().isoformat()
            entry.access_count += 1
            self._persist()
        return entry

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[MemorySearchResult]:
        """Search long-term memory.

        Args:
            query: Search query
            limit: Maximum results
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold

        Returns:
            Search results with relevance scores
        """
        results = []
        query_lower = query.lower()
        query_keywords = set(self._extract_keywords(query))

        # Determine which entries to search
        if memory_types:
            candidate_ids = set()
            for mt in memory_types:
                candidate_ids.update(self._type_index.get(mt, []))
            candidates = [
                self._entries[eid] for eid in candidate_ids if eid in self._entries
            ]
        else:
            candidates = list(self._entries.values())

        for entry in candidates:
            if entry.importance < min_importance:
                continue

            # Calculate relevance
            content_keywords = set(self._extract_keywords(entry.content))

            # Keyword overlap
            overlap = len(query_keywords & content_keywords)
            keyword_score = overlap / max(len(query_keywords), 1)

            # Entity match bonus
            entity_score = 0.0
            if entry.entity_name and entry.entity_name.lower() in query_lower:
                entity_score = 0.5

            # Substring match
            substring_score = 0.3 if query_lower in entry.content.lower() else 0.0

            # Combined score with importance weight
            relevance = (
                keyword_score * 0.4
                + entity_score * 0.2
                + substring_score * 0.2
                + entry.importance * 0.2
            )

            if relevance > 0.1:
                results.append(
                    MemorySearchResult(
                        entry=entry,
                        relevance_score=min(1.0, relevance),
                    )
                )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def search_by_entity(self, entity_name: str) -> List[MemoryEntry]:
        """Search memories by entity name."""
        entry_ids = self._entity_index.get(entity_name.lower(), [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Get memories by type."""
        entry_ids = self._type_index.get(memory_type, [])
        entries = [self._entries[eid] for eid in entry_ids if eid in self._entries]
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:limit]

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id not in self._entries:
            return False

        entry = self._entries.pop(entry_id)

        # Update indexes
        if entry.memory_type in self._type_index:
            self._type_index[entry.memory_type] = [
                eid for eid in self._type_index[entry.memory_type] if eid != entry_id
            ]

        if entry.entity_name:
            entity_key = entry.entity_name.lower()
            if entity_key in self._entity_index:
                self._entity_index[entity_key] = [
                    eid for eid in self._entity_index[entity_key] if eid != entry_id
                ]

        self._persist()
        return True

    def clear(self) -> None:
        """Clear all long-term memory."""
        self._entries.clear()
        self._entity_index.clear()
        self._type_index.clear()
        self._persist()

    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text.lower())
        return words

    def _persist(self) -> None:
        """Persist memory to disk."""
        if not self.persist_dir:
            return

        data = {
            "entries": {k: v.model_dump() for k, v in self._entries.items()},
            "entity_index": dict(self._entity_index),
            "type_index": {k.value: v for k, v in self._type_index.items()},
        }

        file_path = self.persist_dir / "long_term_memory.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """Load memory from disk."""
        if not self.persist_dir:
            return

        file_path = self.persist_dir / "long_term_memory.json"
        if not file_path.exists():
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry_id, entry_data in data.get("entries", {}).items():
                self._entries[entry_id] = MemoryEntry(**entry_data)

            for entity, entry_ids in data.get("entity_index", {}).items():
                self._entity_index[entity] = entry_ids

            for type_str, entry_ids in data.get("type_index", {}).items():
                try:
                    memory_type = MemoryType(type_str)
                    self._type_index[memory_type] = entry_ids
                except ValueError:
                    pass

        except Exception as e:
            print(f"Failed to load long-term memory: {e}")


class WorkingMemory:
    """Working memory for current query context.

    Aggregates relevant information from various sources
    for the current query processing.
    """

    def __init__(self, max_context_tokens: int = 4000):
        """Initialize working memory.

        Args:
            max_context_tokens: Maximum tokens for context
        """
        self.max_context_tokens = max_context_tokens
        self._current_query: Optional[str] = None
        self._conversation_context: List[Message] = []
        self._relevant_memories: List[MemoryEntry] = []
        self._retrieved_docs: List[Any] = []
        self._entities: Dict[str, str] = {}
        self._facts: List[str] = []

    def set_query(self, query: str) -> None:
        """Set the current query."""
        self._current_query = query

    def add_conversation_context(self, messages: List[Message]) -> None:
        """Add conversation history to working memory."""
        self._conversation_context = messages

    def add_memories(self, memories: List[MemoryEntry]) -> None:
        """Add relevant memories to working memory."""
        self._relevant_memories = memories

    def add_documents(self, docs: List[Any]) -> None:
        """Add retrieved documents."""
        self._retrieved_docs = docs

    def add_entity(self, name: str, value: str) -> None:
        """Add an extracted entity."""
        self._entities[name] = value

    def add_fact(self, fact: str) -> None:
        """Add a fact."""
        if fact not in self._facts:
            self._facts.append(fact)

    def get_context_string(self, max_tokens: Optional[int] = None) -> str:
        """Build context string from working memory.

        Args:
            max_tokens: Maximum tokens (uses default if not specified)

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.max_context_tokens
        parts = []
        token_count = 0

        # Add conversation history
        if self._conversation_context:
            history_parts = []
            for msg in self._conversation_context[-6:]:  # Last 6 messages
                role = "用户" if msg.role == MessageRole.USER else "助手"
                history_parts.append(f"{role}: {msg.content}")

            history_text = "\n".join(history_parts)
            history_tokens = self._estimate_tokens(history_text)

            if token_count + history_tokens < max_tokens:
                parts.append(f"对话历史:\n{history_text}")
                token_count += history_tokens

        # Add relevant memories
        if self._relevant_memories:
            memory_parts = []
            for mem in self._relevant_memories[:5]:
                mem_text = f"- [{mem.memory_type.value}] {mem.content}"
                mem_tokens = self._estimate_tokens(mem_text)
                if token_count + mem_tokens < max_tokens:
                    memory_parts.append(mem_text)
                    token_count += mem_tokens

            if memory_parts:
                parts.append("相关记忆:\n" + "\n".join(memory_parts))

        # Add entities
        if self._entities:
            entity_text = "已知实体: " + ", ".join(
                f"{k}={v}" for k, v in self._entities.items()
            )
            entity_tokens = self._estimate_tokens(entity_text)
            if token_count + entity_tokens < max_tokens:
                parts.append(entity_text)
                token_count += entity_tokens

        # Add facts
        if self._facts:
            facts_text = "相关事实:\n" + "\n".join(f"- {f}" for f in self._facts[:5])
            facts_tokens = self._estimate_tokens(facts_text)
            if token_count + facts_tokens < max_tokens:
                parts.append(facts_text)
                token_count += facts_tokens

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear working memory."""
        self._current_query = None
        self._conversation_context = []
        self._relevant_memories = []
        self._retrieved_docs = []
        self._entities = {}
        self._facts = []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5) + int(other_chars / 4)


class MemorySystem:
    """Unified memory system combining all memory types.

    Provides a single interface for:
    - Short-term memory (session context)
    - Long-term memory (persistent knowledge)
    - Working memory (current query context)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        short_term_capacity: int = 50,
        short_term_ttl: int = 60,
        max_context_tokens: int = 4000,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
    ):
        """Initialize the memory system.

        Args:
            persist_dir: Directory for long-term memory persistence
            short_term_capacity: Capacity of short-term memory
            short_term_ttl: TTL for short-term memories in minutes
            max_context_tokens: Maximum tokens for working memory
            embedding_func: Function to generate embeddings
        """
        self.short_term = ShortTermMemory(
            capacity=short_term_capacity,
            ttl_minutes=short_term_ttl,
        )
        self.long_term = LongTermMemory(
            persist_dir=persist_dir,
            embedding_func=embedding_func,
        )
        self.working = WorkingMemory(
            max_context_tokens=max_context_tokens,
        )

    def remember(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        persist: bool = False,
        **kwargs,
    ) -> MemoryEntry:
        """Add a memory to the appropriate store.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            persist: Whether to store in long-term memory
            **kwargs: Additional arguments for the memory entry

        Returns:
            Created MemoryEntry
        """
        if persist or importance >= 0.7:
            return self.long_term.add(
                content=content,
                memory_type=memory_type,
                importance=importance,
                **kwargs,
            )
        else:
            return self.short_term.add(
                content=content,
                memory_type=memory_type,
                importance=importance,
                **kwargs,
            )

    def recall(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
    ) -> List[MemorySearchResult]:
        """Recall relevant memories.

        Args:
            query: Search query
            limit: Maximum results
            memory_types: Filter by memory types
            include_short_term: Include short-term memories
            include_long_term: Include long-term memories

        Returns:
            Combined search results
        """
        results = []

        if include_short_term:
            st_results = self.short_term.search(
                query, limit=limit, memory_types=memory_types
            )
            results.extend(st_results)

        if include_long_term:
            lt_results = self.long_term.search(
                query, limit=limit, memory_types=memory_types
            )
            results.extend(lt_results)

        # Deduplicate and sort
        seen_contents = set()
        unique_results = []
        for r in results:
            if r.entry.content not in seen_contents:
                seen_contents.add(r.entry.content)
                unique_results.append(r)

        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return unique_results[:limit]

    def prepare_context(
        self,
        query: str,
        conversation_messages: Optional[List[Message]] = None,
        retrieved_docs: Optional[List[Any]] = None,
    ) -> str:
        """Prepare context for the current query.

        Args:
            query: Current query
            conversation_messages: Recent conversation messages
            retrieved_docs: Retrieved documents

        Returns:
            Formatted context string
        """
        self.working.clear()
        self.working.set_query(query)

        if conversation_messages:
            self.working.add_conversation_context(conversation_messages)

        # Recall relevant memories
        memories = self.recall(query, limit=5)
        self.working.add_memories([r.entry for r in memories])

        if retrieved_docs:
            self.working.add_documents(retrieved_docs)

        return self.working.get_context_string()

    def extract_and_store_memories(
        self,
        message: str,
        is_user: bool = True,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Extract and store memories from a message.

        Args:
            message: The message to process
            is_user: Whether this is a user message
            conversation_id: Source conversation
            message_id: Source message

        Returns:
            List of extracted memories
        """
        extracted = []

        # Extract preferences (user messages only)
        if is_user:
            preferences = self._extract_preferences(message)
            for pref in preferences:
                entry = self.remember(
                    content=pref,
                    memory_type=MemoryType.PREFERENCE,
                    importance=0.8,
                    persist=True,
                    source_conversation_id=conversation_id,
                    source_message_id=message_id,
                )
                extracted.append(entry)

        # Extract entities
        entities = self._extract_entities(message)
        for name, desc in entities.items():
            entry = self.remember(
                content=desc,
                memory_type=MemoryType.ENTITY,
                importance=0.6,
                persist=True,
                entity_name=name,
                source_conversation_id=conversation_id,
                source_message_id=message_id,
            )
            extracted.append(entry)

        # Store as short-term context
        self.short_term.add(
            content=message,
            memory_type=MemoryType.CONTEXT,
            importance=0.4,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
        )

        return extracted

    def _extract_preferences(self, message: str) -> List[str]:
        """Extract user preferences from message."""
        preferences = []

        # Patterns for preferences
        patterns = [
            (r"我喜欢(.+)", "用户喜欢{}"),
            (r"我不喜欢(.+)", "用户不喜欢{}"),
            (r"我偏好(.+)", "用户偏好{}"),
            (r"我习惯(.+)", "用户习惯{}"),
            (r"我想要(.+)", "用户想要{}"),
            (r"请用(.+)回答", "用户希望用{}格式回答"),
        ]

        for pattern, template in patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if len(match) > 2 and len(match) < 50:
                    preferences.append(template.format(match.strip()))

        return preferences

    def _extract_entities(self, message: str) -> Dict[str, str]:
        """Extract entities from message."""
        entities = {}

        # Simple entity patterns
        patterns = [
            (r"我叫(\S+)", "name", "用户名字是{}"),
            (r"我是(\S+)", "identity", "用户身份是{}"),
            (r"我在(.+?)工作", "workplace", "用户工作地点是{}"),
            (r"我住在(.+)", "location", "用户住在{}"),
        ]

        for pattern, entity_type, template in patterns:
            match = re.search(pattern, message)
            if match:
                value = match.group(1).strip()
                if len(value) > 1 and len(value) < 30:
                    entities[entity_type] = template.format(value)

        return entities

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of memory system state."""
        return {
            "short_term_count": len(self.short_term._entries),
            "long_term_count": len(self.long_term._entries),
            "entity_count": len(self.long_term._entity_index),
            "memory_types": {
                mt.value: len(ids) for mt, ids in self.long_term._type_index.items()
            },
        }
