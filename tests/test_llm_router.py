"""Tests for LLM Router — Haiku default, Sonnet escalation."""

import pytest
from server.routers.llm_router import (
    route_llm, RoutingDecision, HAIKU_MODEL, SONNET_MODEL,
)


class TestDefaultRouting:
    def test_default_is_haiku(self):
        result = route_llm("Hello, how are you today?")
        assert result.model == HAIKU_MODEL
        assert result.reason is None

    def test_normal_conversation_uses_haiku(self):
        result = route_llm("What's the weather like?")
        assert result.model == HAIKU_MODEL

    def test_short_turns_use_haiku(self):
        result = route_llm("Yes, I had breakfast.", turn_count=3)
        assert result.model == HAIKU_MODEL


class TestOnboardingEscalation:
    def test_onboarding_active_triggers_sonnet(self):
        result = route_llm("My name is Margaret.", onboarding_active=True)
        assert result.model == SONNET_MODEL
        assert result.reason == "onboarding_active"

    def test_onboarding_inactive_stays_haiku(self):
        result = route_llm("My name is Margaret.", onboarding_active=False)
        assert result.model == HAIKU_MODEL


class TestEmotionalKeywords:
    @pytest.mark.parametrize("keyword", [
        "grief", "loss", "scared", "lonely", "miss", "died", "worried",
        "confused", "pain", "afraid", "crying", "upset", "sad", "hurting",
    ])
    def test_emotional_keyword_triggers_sonnet(self, keyword):
        result = route_llm(f"I feel so {keyword} today.")
        assert result.model == SONNET_MODEL
        assert "emotional_keyword" in result.reason

    def test_emotional_keyword_case_insensitive(self):
        result = route_llm("I'm so LONELY without her.")
        assert result.model == SONNET_MODEL

    def test_emotional_keyword_in_sentence(self):
        result = route_llm("My husband died three years ago.")
        assert result.model == SONNET_MODEL
        assert "died" in result.reason

    def test_no_false_positive_on_similar_words(self):
        result = route_llm("I had a great day at the park.")
        assert result.model == HAIKU_MODEL


class TestConversationDepth:
    def test_depth_under_10_uses_haiku(self):
        result = route_llm("Tell me more", turn_count=10)
        assert result.model == HAIKU_MODEL

    def test_depth_over_10_triggers_sonnet(self):
        result = route_llm("Tell me more", turn_count=11)
        assert result.model == SONNET_MODEL
        assert result.reason == "conversation_depth"


class TestMemoryCategoryEscalation:
    @pytest.mark.parametrize("category", ["stories", "meaning", "faith", "emotions"])
    def test_sonnet_memory_categories_trigger(self, category):
        result = route_llm("Yes, that's right.", last_memory_categories=[category])
        assert result.model == SONNET_MODEL
        assert f"memory_category:{category}" == result.reason

    def test_non_sonnet_categories_stay_haiku(self):
        result = route_llm("Yes, that's right.", last_memory_categories=["health", "routine"])
        assert result.model == HAIKU_MODEL

    def test_empty_categories_stay_haiku(self):
        result = route_llm("Yes.", last_memory_categories=[])
        assert result.model == HAIKU_MODEL


class TestEscalationPriority:
    def test_onboarding_takes_priority(self):
        """Onboarding check happens before keyword check."""
        result = route_llm("I'm sad", onboarding_active=True)
        assert result.reason == "onboarding_active"

    def test_escalation_logging_has_reason(self):
        result = route_llm("I miss my wife", turn_count=15)
        # Emotional keyword should fire before depth
        assert result.model == SONNET_MODEL
        assert result.reason is not None
