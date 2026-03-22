"""Tests for Context Builder."""

import pytest
from server.context import (
    build_context, format_memory_summary, format_conversation_history,
    get_time_awareness, load_system_prompt,
)


class TestSystemPrompt:
    def test_system_prompt_loads(self):
        prompt = load_system_prompt()
        assert "EverNear" in prompt
        assert len(prompt) > 100

    def test_system_prompt_contains_behavioral_rules(self):
        prompt = load_system_prompt()
        assert "never" in prompt.lower() or "NEVER" in prompt


class TestMemorySummary:
    def test_empty_memories(self):
        assert format_memory_summary([]) == ""

    def test_single_memory(self):
        memories = [{"category": "family", "content": "Has a daughter named Sarah"}]
        result = format_memory_summary(memories)
        assert "Sarah" in result
        assert "Family" in result

    def test_multiple_categories(self):
        memories = [
            {"category": "family", "content": "Has a daughter named Sarah"},
            {"category": "health", "content": "Takes blood pressure medication"},
            {"category": "interests", "content": "Loves crossword puzzles"},
        ]
        result = format_memory_summary(memories)
        assert "Family" in result
        assert "Health" in result
        assert "Interests" in result

    def test_caps_per_category(self):
        """Should cap at 5 items per category to keep token budget."""
        memories = [
            {"category": "family", "content": f"Family fact {i}"}
            for i in range(10)
        ]
        result = format_memory_summary(memories)
        # Should only include up to 5
        assert "Family fact 0" in result
        assert "Family fact 4" in result
        # fact 5+ should be excluded
        assert "Family fact 5" not in result


class TestConversationHistory:
    def test_empty_history(self):
        assert format_conversation_history([]) == []

    def test_preserves_role_and_content(self):
        msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        result = format_conversation_history(msgs)
        assert result[0]["role"] == "user"
        assert result[1]["content"] == "Hi there"

    def test_truncates_to_max_turns(self):
        # 20 messages = 10 turns, max_turns=3 → last 6 messages
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(20)
        ]
        result = format_conversation_history(msgs, max_turns=3)
        assert len(result) == 6
        assert result[0]["content"] == "msg14"


class TestTimeAwareness:
    def test_returns_string(self):
        result = get_time_awareness()
        assert isinstance(result, str)
        assert len(result) > 10


class TestBuildContext:
    def test_includes_user_name(self):
        system, history = build_context(
            user_name="Margaret",
            memories=[],
            conversation_history=[],
        )
        assert "Margaret" in system

    def test_includes_memories(self):
        system, history = build_context(
            user_name="Bob",
            memories=[{"category": "family", "content": "Has a son named Tom"}],
            conversation_history=[],
        )
        assert "Tom" in system

    def test_includes_onboarding_context(self):
        system, history = build_context(
            user_name="Alice",
            memories=[],
            conversation_history=[],
            onboarding_state={"current_section": "welcome", "completed": False},
        )
        assert "ONBOARDING" in system or "onboarding" in system.lower()

    def test_no_onboarding_when_completed(self):
        system, history = build_context(
            user_name="Alice",
            memories=[],
            conversation_history=[],
            onboarding_state={"current_section": "welcome", "completed": True},
        )
        # Should not include onboarding-specific section
        assert "Current Section" not in system

    def test_returning_user_with_memories_gets_warm_greeting(self):
        system, _ = build_context(
            user_name="Margaret",
            memories=[{"category": "family", "content": "Has a daughter named Sarah"}],
            conversation_history=[],
            onboarding_state={"completed": True},
        )
        assert "talked to you before" in system
        assert "remember things about them" in system

    def test_returning_user_without_memories_gets_honest_greeting(self):
        system, _ = build_context(
            user_name="Bob",
            memories=[],
            conversation_history=[],
            onboarding_state={"completed": True},
        )
        assert "talked to you before" in system
        assert "don't remember specifics" in system

    def test_new_user_no_returning_greeting(self):
        system, _ = build_context(
            user_name="Alice",
            memories=[],
            conversation_history=[],
            onboarding_state={"current_section": "welcome", "completed": False},
        )
        assert "talked to you before" not in system

    def test_context_is_reasonable_size(self):
        """Context should stay manageable for 4K token budget."""
        memories = [
            {"category": "family", "content": f"Family member {i}"}
            for i in range(10)
        ]
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"User message {i}"})
            history.append({"role": "assistant", "content": f"Assistant response {i}"})
        system, hist = build_context("Margaret", memories, history, max_turns=8)
        # Rough token estimate: 1 token ≈ 4 chars
        total_chars = len(system) + sum(len(m["content"]) for m in hist)
        estimated_tokens = total_chars / 4
        assert estimated_tokens < 6000  # Allow some headroom above 4K
