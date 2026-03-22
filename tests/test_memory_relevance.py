"""Tests for memory relevance, deduplication, recency scoring, and correction logic."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from server.memory import (
    _normalize_text, _fuzzy_match, _is_correction, _recency_score,
    deduplicate_and_store, get_user_memories,
)
from server.context import (
    format_memory_summary, _content_words, _score_relevance, build_context,
)


class TestRecencyScoring:
    def test_today_scores_one(self):
        now = datetime.now(timezone.utc)
        score = _recency_score(now.isoformat(), now)
        assert score == 1.0

    def test_30_days_ago_scores_floor(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=30)).isoformat()
        score = _recency_score(old, now)
        assert score == pytest.approx(0.3, abs=0.01)

    def test_60_days_ago_scores_floor(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()
        score = _recency_score(old, now)
        assert score == 0.3

    def test_15_days_ago_midpoint(self):
        now = datetime.now(timezone.utc)
        mid = (now - timedelta(days=15)).isoformat()
        score = _recency_score(mid, now)
        assert 0.5 < score < 0.8

    def test_no_timestamp_returns_floor(self):
        assert _recency_score(None) == 0.3

    def test_recency_ordering(self):
        """Recent memories should score higher than old ones with same importance."""
        now = datetime.now(timezone.utc)
        recent = _recency_score((now - timedelta(days=1)).isoformat(), now)
        old = _recency_score((now - timedelta(days=25)).isoformat(), now)
        assert recent > old


class TestFuzzyDedup:
    def test_exact_match(self):
        assert _fuzzy_match("My cat is Whiskers", "My cat is Whiskers")

    def test_near_duplicate_substring(self):
        assert _fuzzy_match("My cat is Whiskers", "My cat's name is Whiskers")

    def test_high_word_overlap(self):
        assert _fuzzy_match("She has a daughter named Sarah", "Has a daughter named Sarah")

    def test_different_content_no_match(self):
        assert not _fuzzy_match("Likes chocolate ice cream", "Works at the hospital downtown")

    def test_empty_strings(self):
        assert not _fuzzy_match("", "Something")

    def test_normalize_strips_punctuation(self):
        norm = _normalize_text("Hello, World! It's a test.")
        assert "," not in norm
        assert "!" not in norm


class TestCorrectionDetection:
    def test_actually_pattern(self):
        assert _is_correction("Actually, her name is Mittens")

    def test_no_its_pattern(self):
        assert _is_correction("No, it's Mittens not Whiskers")

    def test_i_meant_pattern(self):
        assert _is_correction("I meant to say Tuesday")

    def test_correction_keyword(self):
        assert _is_correction("Correction: it's on Wednesday")

    def test_normal_message_not_correction(self):
        assert not _is_correction("My daughter visited today")

    def test_case_insensitive(self):
        assert _is_correction("ACTUALLY, it's different")


class TestRelevanceScoring:
    def test_relevant_memory(self):
        user_words = _content_words("How is your daughter Sarah?")
        score = _score_relevance("Has a daughter named Sarah", user_words)
        assert score > 0

    def test_irrelevant_memory(self):
        user_words = _content_words("What's the weather like?")
        score = _score_relevance("Takes blood pressure medication daily", user_words)
        assert score == 0

    def test_stop_words_excluded(self):
        words = _content_words("the is a an to for")
        assert len(words) == 0

    def test_empty_message(self):
        score = _score_relevance("Has a cat", set())
        assert score == 0


class TestFormatMemorySummaryRelevance:
    def test_relevant_memories_promoted(self):
        memories = [
            {"category": "family", "content": "Has a daughter named Sarah"},
            {"category": "health", "content": "Takes blood pressure medication"},
        ]
        result = format_memory_summary(memories, user_message="How is Sarah doing?")
        # Sarah-related memory should appear first with "You know that" prefix
        sarah_pos = result.find("Sarah")
        medication_pos = result.find("medication")
        assert sarah_pos < medication_pos

    def test_background_capped_at_2_when_message_provided(self):
        memories = [
            {"category": "family", "content": f"Family fact {i}"}
            for i in range(10)
        ]
        result = format_memory_summary(memories, user_message="Tell me about weather")
        # All are background (no overlap with "weather"), capped at 2
        count = sum(1 for i in range(10) if f"Family fact {i}" in result)
        assert count <= 2

    def test_no_message_uses_original_cap(self):
        memories = [
            {"category": "family", "content": f"Family fact {i}"}
            for i in range(10)
        ]
        result = format_memory_summary(memories, user_message="")
        count = sum(1 for i in range(10) if f"Family fact {i}" in result)
        assert count <= 5

    def test_token_budget_respected(self):
        memories = [
            {"category": f"cat{i}", "content": "A" * 500}
            for i in range(20)
        ]
        result = format_memory_summary(memories, token_budget=100)
        # 100 tokens * 4 chars = 400 char budget — should be trimmed
        assert len(result) < 800


class TestTokenBudgetInBuildContext:
    def test_large_memory_set_trimmed(self):
        """build_context should handle large memory sets without exceeding budget."""
        memories = [
            {"category": "family", "content": f"Long family fact number {i} with extra detail " * 3}
            for i in range(50)
        ]
        system, _ = build_context(
            user_name="Test",
            memories=memories,
            conversation_history=[],
            user_message="Hello",
        )
        # Should stay under reasonable size
        assert len(system) < 12000  # ~3000 tokens


class TestDeduplicateAndStoreFuzzy:
    @pytest.mark.asyncio
    async def test_fuzzy_duplicate_skipped(self):
        """Near-duplicate should be skipped, not stored again."""
        mock_select = MagicMock()
        mock_select.data = [{"id": "existing-1", "content": "My cat is Whiskers", "category": "family"}]
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_select

        with patch("server.memory.get_service_client") as mock_db:
            mock_db.return_value.table.return_value = mock_table
            memories = [{"category": "family", "content": "My cat's name is Whiskers", "importance": 3}]
            result = await deduplicate_and_store(memories, "user123")
            assert result == 0

    @pytest.mark.asyncio
    async def test_correction_supersedes(self):
        """Correction should deactivate old memory and store new one."""
        mock_select = MagicMock()
        # Old: "Daughter Sarah lives in Austin Texas" — correction with high overlap
        mock_select.data = [{"id": "existing-1", "content": "Daughter Sarah lives in Austin Texas", "category": "family"}]
        mock_insert = MagicMock()
        mock_insert.data = [{"id": "new-1"}]
        mock_update = MagicMock()
        mock_update.data = [{}]

        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_select
        mock_table.insert.return_value.execute.return_value = mock_insert
        mock_table.update.return_value.eq.return_value.execute.return_value = mock_update

        with patch("server.memory.get_service_client") as mock_db:
            mock_db.return_value.table.return_value = mock_table
            memories = [{"category": "family", "content": "Daughter Sarah lives in Houston Texas", "importance": 3}]
            result = await deduplicate_and_store(
                memories, "user123",
                user_message="Actually, Sarah lives in Houston not Austin"
            )
            assert result == 1
            # Verify update was called to deactivate old memory
            mock_table.update.assert_called()
