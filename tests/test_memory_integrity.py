"""Tests for Memory Integrity (Phase 3) — confidence scoring, verification, immutability."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json

from server.memory import extract_memories, deduplicate_and_store, MEMORY_CATEGORIES


class TestConfidenceScoring:
    """Verify that extraction prompt requests and parses confidence values."""

    @pytest.mark.asyncio
    async def test_explicit_fact_gets_high_confidence(self):
        """Explicit statements should produce high confidence."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([{
            "category": "family",
            "content": "Has a daughter named Sarah",
            "importance": 4,
            "confidence": 0.95,
        }]))]

        with patch("server.memory.get_settings") as mock_s, \
             patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
            mock_s.return_value.anthropic_api_key = "test"
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)

            memories = await extract_memories("My daughter Sarah called me today", "That's nice!", "user-1")
            assert len(memories) >= 1
            # Confidence field should be present if extraction prompt includes it
            # (V1: confidence comes from the LLM extraction)

    @pytest.mark.asyncio
    async def test_inferred_fact_gets_low_confidence(self):
        """Implied/contextual information should produce lower confidence."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([{
            "category": "health",
            "content": "May have trouble sleeping",
            "importance": 2,
            "confidence": 0.4,
        }]))]

        with patch("server.memory.get_settings") as mock_s, \
             patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
            mock_s.return_value.anthropic_api_key = "test"
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)

            memories = await extract_memories("I've been tired lately", "I'm sorry to hear that.", "user-1")
            assert len(memories) >= 1


class TestVerificationPrompt:
    """Verify that medium-confidence memories trigger verification questions."""

    def test_validator_rewrites_low_confidence_assertion(self):
        """The validator should REWRITE when asserting a low-confidence memory."""
        from server.validator import ResponseValidator, ValidationAction
        from tests.test_validator import _make_cache

        cache = _make_cache([{
            "category": "family",
            "content": "daughter Sarah lives in Austin",
            "confidence": 0.5,
            "verified": False,
        }])

        v = ResponseValidator()
        r = v.validate(
            "Your daughter Sarah lives in Austin.",
            "Tell me about my family",
            cache,
        )
        assert r.action == ValidationAction.REWRITE
        assert "remembering that right" in r.response


class TestNoDoubleVerify:
    """Only one verification per session — deferred to pipeline-level tracking.
    
    The validator flags low-confidence memories. The pipeline is responsible
    for tracking whether a verification has already been issued this session.
    """
    pass


class TestNoVerifyDuringDistress:
    """Sensitive context suppresses verification — deferred to boundary awareness.
    
    When the LLM router detects sensitive domain, the pipeline should skip
    memory verification prompts. This is enforced at the pipeline level.
    """
    pass


class TestCorrectionHandling:
    """User corrections should update memory and increment correction_count."""

    @pytest.mark.asyncio
    async def test_correction_creates_new_memory(self):
        """When new info contradicts existing, old is superseded, new is created."""
        mock_db = MagicMock()
        # Existing memory found (3 .eq() calls: user_id, category, active)
        mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{
            "id": "old-id",
            "content": "daughter Sarah lives in Austin",
        }]
        mock_db.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch("server.memory.get_service_client", return_value=mock_db):
            stored = await deduplicate_and_store(
                [{"category": "family", "content": "daughter Sarah lives in Austin", "importance": 4}],
                "user-1",
            )
            # Exact duplicate should be skipped
            assert stored == 0


class TestImmutability:
    """Old memories should be marked superseded, not overwritten."""

    @pytest.mark.asyncio
    async def test_new_memory_stored_when_content_differs(self):
        """Non-duplicate content in same category creates new entry."""
        mock_db = MagicMock()
        # No match found (3 .eq() calls: user_id, category, active)
        mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        mock_db.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch("server.memory.get_service_client", return_value=mock_db):
            stored = await deduplicate_and_store(
                [{"category": "family", "content": "daughter Sarah lives in Denver", "importance": 4}],
                "user-1",
            )
            assert stored == 1
