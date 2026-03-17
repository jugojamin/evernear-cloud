"""Tests for Conversation Boundary Awareness (Phase 4)."""

import pytest
from server.routers.llm_router import (
    route_llm,
    SONNET_MODEL,
    HAIKU_MODEL,
    EMOTIONAL_KEYWORDS,
)


class TestSensitiveDomainDetection:
    """Each sensitive domain keyword should trigger Sonnet routing."""

    @pytest.mark.parametrize("keyword", [
        "grief", "loss", "scared", "lonely", "miss", "died",
        "worried", "confused", "pain", "afraid", "crying",
        "upset", "sad", "hurting", "passed away", "frightened",
        "anxious", "depressed",
    ])
    def test_emotional_keyword_triggers_sonnet(self, keyword):
        result = route_llm(transcript=f"I've been feeling {keyword} lately")
        assert result.model == SONNET_MODEL
        assert result.reason is not None
        assert "emotional_keyword" in result.reason


class TestSonnetRouting:
    def test_onboarding_triggers_sonnet(self):
        result = route_llm(transcript="Hello", onboarding_active=True)
        assert result.model == SONNET_MODEL
        assert result.reason == "onboarding_active"

    def test_conversation_depth_triggers_sonnet(self):
        result = route_llm(transcript="Tell me more", turn_count=11)
        assert result.model == SONNET_MODEL
        assert result.reason == "conversation_depth"

    def test_memory_category_triggers_sonnet(self):
        result = route_llm(
            transcript="Tell me more",
            last_memory_categories=["stories"],
        )
        assert result.model == SONNET_MODEL
        assert "memory_category" in result.reason

    def test_benign_input_uses_haiku(self):
        result = route_llm(transcript="What's for lunch?")
        assert result.model == HAIKU_MODEL
        assert result.reason is None


class TestResponseLength:
    """Sensitive mode should produce shorter responses.
    
    This is enforced via the system prompt's SENSITIVE MODE injection
    in the context builder. Full integration test deferred to pipeline
    integration testing with real LLM calls.
    """
    pass


class TestPacingDelay:
    """Sensitive mode should add a configurable delay before TTS.
    
    This is a pipeline-level behavior — the delay is added in the
    voice pipeline path, not in the text-mode path. Deferred to
    voice pipeline integration testing.
    """
    pass
