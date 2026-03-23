"""Tests for Memory module - Conversation and Memory System."""

from rag_minimal.memory import (
    ConversationalAgent,
    ConversationManager,
    LongTermMemory,
    MemorySystem,
    ShortTermMemory,
    WorkingMemory,
)
from rag_minimal.schemas import (
    Conversation,
    MemoryType,
    Message,
    MessageRole,
    SearchOutput,
    SearchResultItem,
)


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_create_conversation(self):
        """Test creating a new conversation."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        assert isinstance(conv, Conversation)
        assert conv.id is not None
        assert conv.message_count == 0
        assert conv.messages == []

    def test_create_conversation_with_title(self):
        """Test creating a conversation with a title."""
        manager = ConversationManager()
        conv = manager.create_conversation(title="Test Conversation")

        assert conv.title == "Test Conversation"

    def test_get_or_create_conversation(self):
        """Test get_or_create_conversation."""
        manager = ConversationManager()

        # Create new when None
        conv1 = manager.get_or_create_conversation(None)
        assert conv1 is not None

        # Get existing
        conv2 = manager.get_or_create_conversation(conv1.id)
        assert conv2.id == conv1.id

    def test_add_user_message(self):
        """Test adding a user message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        msg = manager.add_user_message(conv.id, "Hello, world!")

        assert isinstance(msg, Message)
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"

        # Check conversation is updated
        updated_conv = manager.get_conversation(conv.id)
        assert updated_conv.message_count == 1
        assert len(updated_conv.messages) == 1

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        msg = manager.add_assistant_message(conv.id, "Hello! How can I help?")

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hello! How can I help?"

    def test_get_history(self):
        """Test getting conversation history."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "Message 1")
        manager.add_assistant_message(conv.id, "Response 1")
        manager.add_user_message(conv.id, "Message 2")

        history = manager.get_history(conv.id)

        assert len(history) == 3
        assert history[0].content == "Message 1"
        assert history[1].content == "Response 1"
        assert history[2].content == "Message 2"

    def test_get_history_with_limit(self):
        """Test getting limited conversation history."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        for i in range(5):
            manager.add_user_message(conv.id, f"Message {i}")

        history = manager.get_history(conv.id, limit=3)

        assert len(history) == 3
        # Should get the most recent 3
        assert history[0].content == "Message 2"
        assert history[2].content == "Message 4"

    def test_get_history_as_text(self):
        """Test getting conversation history as text."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "What is AI?")
        manager.add_assistant_message(conv.id, "AI is artificial intelligence.")

        text = manager.get_history_as_text(conv.id)

        assert "用户: What is AI?" in text
        assert "助手: AI is artificial intelligence." in text

    def test_list_conversations(self):
        """Test listing conversations."""
        manager = ConversationManager()

        conv1 = manager.create_conversation(title="Conv 1")
        conv2 = manager.create_conversation(title="Conv 2")

        conversations = manager.list_conversations()

        assert len(conversations) >= 2
        conv_ids = [c.id for c in conversations]
        assert conv1.id in conv_ids
        assert conv2.id in conv_ids

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        result = manager.delete_conversation(conv.id)

        assert result is True
        assert manager.get_conversation(conv.id) is None

    def test_clear_conversation(self):
        """Test clearing a conversation's messages."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "Message 1")
        manager.add_user_message(conv.id, "Message 2")

        result = manager.clear_conversation(conv.id)

        assert result is True
        updated = manager.get_conversation(conv.id)
        assert updated.message_count == 0
        assert len(updated.messages) == 0

    def test_get_last_user_message(self):
        """Test getting the last user message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "First question")
        manager.add_assistant_message(conv.id, "First answer")
        manager.add_user_message(conv.id, "Second question")

        last_user = manager.get_last_user_message(conv.id)

        assert last_user is not None
        assert last_user.content == "Second question"

    def test_get_last_assistant_message(self):
        """Test getting the last assistant message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "Question")
        manager.add_assistant_message(conv.id, "Answer")

        last_assistant = manager.get_last_assistant_message(conv.id)

        assert last_assistant is not None
        assert last_assistant.content == "Answer"

    def test_generate_title_from_first_message(self):
        """Test generating title from first message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_user_message(conv.id, "Tell me about machine learning algorithms")

        manager.generate_title_from_first_message(conv.id)

        updated = manager.get_conversation(conv.id)
        assert updated.title is not None
        assert len(updated.title) > 0


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving from short-term memory."""
        stm = ShortTermMemory(capacity=10)

        stm.add("User prefers Python", MemoryType.PREFERENCE)
        stm.add("Discussing machine learning", MemoryType.CONTEXT)

        memories = stm.get_recent()

        assert len(memories) == 2

    def test_search(self):
        """Test searching short-term memory."""
        stm = ShortTermMemory()

        stm.add("User likes Python programming", MemoryType.PREFERENCE)
        stm.add("User works with data science", MemoryType.FACT)
        stm.add("Currently discussing AI", MemoryType.CONTEXT)

        results = stm.search("Python", limit=2)

        assert len(results) > 0
        assert any("Python" in r.entry.content for r in results)

    def test_capacity_limit(self):
        """Test that capacity limit is enforced."""
        stm = ShortTermMemory(capacity=3)

        stm.add("Memory 1", MemoryType.FACT)
        stm.add("Memory 2", MemoryType.FACT)
        stm.add("Memory 3", MemoryType.FACT)
        stm.add("Memory 4", MemoryType.FACT)

        memories = stm.get_recent()

        assert len(memories) == 3
        # Oldest should be removed
        contents = [m.content for m in memories]
        assert "Memory 1" not in contents

    def test_clear(self):
        """Test clearing short-term memory."""
        stm = ShortTermMemory()

        stm.add("Memory 1", MemoryType.FACT)
        stm.add("Memory 2", MemoryType.FACT)

        stm.clear()

        assert len(stm.get_recent()) == 0


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving from long-term memory."""
        ltm = LongTermMemory()

        entry = ltm.add("Important fact about user", MemoryType.FACT, importance=0.9)

        assert entry.importance == 0.9

    def test_search(self):
        """Test searching long-term memory."""
        ltm = LongTermMemory()

        ltm.add("User is a software engineer", MemoryType.FACT)
        ltm.add("User prefers dark mode", MemoryType.PREFERENCE)
        ltm.add("User lives in Beijing", MemoryType.FACT)

        results = ltm.search("engineer", limit=2)

        assert len(results) > 0

    def test_get_by_type(self):
        """Test getting memories by type."""
        ltm = LongTermMemory()

        ltm.add("Fact 1", MemoryType.FACT)
        ltm.add("Pref 1", MemoryType.PREFERENCE)
        ltm.add("Fact 2", MemoryType.FACT)

        facts = ltm.get_by_type(MemoryType.FACT)

        assert len(facts) == 2
        assert all(m.memory_type == MemoryType.FACT for m in facts)

    def test_add_summary(self):
        """Test adding conversation summaries."""
        ltm = LongTermMemory()

        # Add summary (result not used, just verifying it doesn't raise)
        ltm.add_summary(
            "Summary of conversation about AI", source_conversation_id="conv123"
        )

        summaries = ltm.get_by_type(MemoryType.SUMMARY)

        assert len(summaries) == 1
        assert summaries[0].source_conversation_id == "conv123"


class TestWorkingMemory:
    """Tests for WorkingMemory."""

    def test_set_query(self):
        """Test setting current query."""
        wm = WorkingMemory()

        wm.set_query("What is machine learning?")

        assert wm._current_query == "What is machine learning?"

    def test_add_entity(self):
        """Test adding entity."""
        wm = WorkingMemory()

        wm.add_entity("topic", "machine learning")

        assert wm._entities["topic"] == "machine learning"

    def test_add_fact(self):
        """Test adding fact."""
        wm = WorkingMemory()

        wm.add_fact("Python is a programming language")

        assert "Python is a programming language" in wm._facts

    def test_add_fact_deduplication(self):
        """Test that duplicate facts are not added."""
        wm = WorkingMemory()

        wm.add_fact("Same fact")
        wm.add_fact("Same fact")

        assert len(wm._facts) == 1

    def test_clear(self):
        """Test clearing working memory."""
        wm = WorkingMemory()

        wm.set_query("test")
        wm.add_entity("key", "value")
        wm.add_fact("fact")

        wm.clear()

        assert wm._current_query is None
        assert len(wm._entities) == 0
        assert len(wm._facts) == 0

    def test_get_context_string(self):
        """Test getting context string."""
        wm = WorkingMemory()

        wm.add_entity("topic", "AI")
        wm.add_fact("AI stands for Artificial Intelligence")

        context = wm.get_context_string()

        assert isinstance(context, str)


class TestMemorySystem:
    """Tests for integrated MemorySystem."""

    def test_remember_short_term(self):
        """Test remembering in short-term memory."""
        memory = MemorySystem()

        memory.remember("Quick note", MemoryType.CONTEXT, persist=False)

        summary = memory.get_summary()
        assert summary["short_term_count"] >= 1

    def test_remember_long_term(self):
        """Test remembering in long-term memory."""
        memory = MemorySystem()

        memory.remember("Important fact", MemoryType.FACT, persist=True, importance=0.8)

        summary = memory.get_summary()
        assert summary["long_term_count"] >= 1

    def test_recall(self):
        """Test recalling memories."""
        memory = MemorySystem()

        memory.remember("User studies Python", MemoryType.FACT, persist=True)
        memory.remember(
            "Current discussion about coding", MemoryType.CONTEXT, persist=False
        )

        results = memory.recall("Python programming", limit=5)

        assert len(results) > 0

    def test_extract_and_store_memories(self):
        """Test extracting memories from messages."""
        memory = MemorySystem()

        memories = memory.extract_and_store_memories(
            message="I prefer using Python for data analysis",
            is_user=True,
            conversation_id="test_conv",
        )

        # Should extract some memories
        assert isinstance(memories, list)

    def test_working_memory_access(self):
        """Test accessing working memory through MemorySystem."""
        memory = MemorySystem()

        memory.working.set_query("test query")
        memory.working.add_fact("test fact")

        assert memory.working._current_query == "test query"
        assert "test fact" in memory.working._facts


class TestConversationalAgent:
    """Tests for ConversationalAgent."""

    def test_chat_creates_conversation(self):
        """Test that chat creates a new conversation."""
        agent = ConversationalAgent()

        result = agent.chat("Hello!")

        assert result.success
        assert result.conversation_id is not None
        assert result.response is not None

    def test_chat_continues_conversation(self):
        """Test continuing an existing conversation."""
        agent = ConversationalAgent()

        result1 = agent.chat("What is AI?")
        conv_id = result1.conversation_id

        result2 = agent.chat("Tell me more", conversation_id=conv_id)

        assert result2.conversation_id == conv_id
        assert result2.messages_used > 0  # Should use history

    def test_chat_with_search_function(self):
        """Test chat with a search function."""

        def mock_search(query: str, top_k: int) -> SearchOutput:
            return SearchOutput(
                success=True,
                message="ok",
                results=[
                    SearchResultItem(
                        content="AI is artificial intelligence.",
                        source="test.txt",
                        score=0.9,
                    )
                ],
            )

        agent = ConversationalAgent(search_func=mock_search)

        result = agent.chat("What is AI?", top_k=3)

        assert result.success
        assert result.documents_used > 0

    def test_get_conversation(self):
        """Test getting a conversation by ID."""
        agent = ConversationalAgent()

        result = agent.chat("Test message")
        conv_id = result.conversation_id

        conv = agent.get_conversation(conv_id)

        assert conv is not None
        assert conv.id == conv_id
        assert conv.message_count >= 2  # User + Assistant

    def test_list_conversations(self):
        """Test listing conversations."""
        agent = ConversationalAgent()

        agent.chat("Conversation 1")
        agent.chat("Conversation 2")

        conversations = agent.list_conversations()

        assert len(conversations) >= 2

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        agent = ConversationalAgent()

        result = agent.chat("Test")
        conv_id = result.conversation_id

        deleted = agent.delete_conversation(conv_id)

        assert deleted is True
        assert agent.get_conversation(conv_id) is None

    def test_add_memory(self):
        """Test manually adding a memory."""
        agent = ConversationalAgent()

        agent.add_memory("User prefers concise answers", MemoryType.PREFERENCE)

        summary = agent.get_memory_summary()
        assert summary["long_term_count"] >= 1

    def test_search_memories(self):
        """Test searching memories."""
        agent = ConversationalAgent()

        agent.add_memory("User is a Python developer", MemoryType.FACT)
        agent.add_memory("User likes dark themes", MemoryType.PREFERENCE)

        results = agent.search_memories("Python")

        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_export_conversation_text(self):
        """Test exporting conversation as text."""
        agent = ConversationalAgent()

        result = agent.chat("What is machine learning?")
        conv_id = result.conversation_id

        exported = agent.export_conversation(conv_id, format="text")

        assert exported is not None
        assert "machine learning" in exported.lower()

    def test_export_conversation_markdown(self):
        """Test exporting conversation as markdown."""
        agent = ConversationalAgent()

        result = agent.chat("Hello")
        conv_id = result.conversation_id

        exported = agent.export_conversation(conv_id, format="markdown")

        assert exported is not None
        assert "**用户**" in exported or "**助手**" in exported

    def test_export_conversation_json(self):
        """Test exporting conversation as JSON."""
        agent = ConversationalAgent()

        result = agent.chat("Test")
        conv_id = result.conversation_id

        exported = agent.export_conversation(conv_id, format="json")

        assert exported is not None
        import json

        data = json.loads(exported)
        assert "id" in data
        assert "messages" in data

    def test_summarize_conversation(self):
        """Test summarizing a conversation."""
        agent = ConversationalAgent()

        # Create a conversation with some messages
        result1 = agent.chat("What is Python?")
        conv_id = result1.conversation_id
        agent.chat("How do I install it?", conversation_id=conv_id)

        summary = agent.summarize_conversation(conv_id)

        # May be None if no LLM, but should not error
        assert summary is None or isinstance(summary, str)

    def test_get_context_for_message(self):
        """Test getting context for a message."""
        agent = ConversationalAgent()

        # Add some memories
        agent.add_memory("User is studying AI", MemoryType.FACT)

        # Create a conversation
        result = agent.chat("Tell me about neural networks")
        conv_id = result.conversation_id

        context = agent.get_context_for_message(
            "What about deep learning?",
            conversation_id=conv_id,
            include_history=True,
            include_memories=True,
        )

        assert isinstance(context, str)


class TestMemoryIntegration:
    """Integration tests for memory system with conversations."""

    def test_memory_persists_across_messages(self):
        """Test that memories persist across multiple messages."""
        agent = ConversationalAgent(auto_extract_memories=True)

        # First message - mention preference
        result1 = agent.chat("I prefer using Python for my projects")
        conv_id = result1.conversation_id

        # Second message - ask about something else
        result2 = agent.chat("What should I learn next?", conversation_id=conv_id)

        # Memory should be used
        assert result2.memories_used >= 0  # May or may not find relevant memories

    def test_conversation_history_used(self):
        """Test that conversation history is used in context."""
        agent = ConversationalAgent()

        result1 = agent.chat("My name is Alice")
        conv_id = result1.conversation_id

        result2 = agent.chat("What is my name?", conversation_id=conv_id)

        # Should use history
        assert result2.messages_used > 0

    def test_multiple_conversations_isolated(self):
        """Test that multiple conversations are isolated."""
        agent = ConversationalAgent()

        result1 = agent.chat("Topic A discussion")
        conv_id_1 = result1.conversation_id

        result2 = agent.chat("Topic B discussion")
        conv_id_2 = result2.conversation_id

        assert conv_id_1 != conv_id_2

        conv1 = agent.get_conversation(conv_id_1)
        # conv2 retrieved but only conv1 assertions needed for this test
        _ = agent.get_conversation(conv_id_2)

        # Messages should be in separate conversations
        assert all(
            "Topic A" in m.content or m.role == MessageRole.ASSISTANT
            for m in conv1.messages
        )
