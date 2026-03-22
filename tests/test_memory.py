"""Tests for Memory Extractor — extraction, dedup, categories."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from server.memory import (
    extract_memories, deduplicate_and_store, process_turn_memories,
    get_user_memories, MEMORY_CATEGORIES,
)


class TestMemoryCategories:
    def test_all_11_categories_defined(self):
        expected = {
            "family", "health", "preferences", "stories", "emotions",
            "meaning", "culture", "faith", "interests", "caregivers", "routine",
        }
        assert set(MEMORY_CATEGORIES) == expected

    def test_category_count(self):
        assert len(MEMORY_CATEGORIES) == 11


class TestExtractMemories:
    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self):
        with patch("server.memory.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = ""
            result = await extract_memories("Hello", "Hi there", "user123")
            assert result == []

    @pytest.mark.asyncio
    async def test_handles_extraction_error_gracefully(self):
        with patch("server.memory.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "test-key"
            with patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
                mock_client.return_value.messages.create = AsyncMock(
                    side_effect=Exception("API error")
                )
                result = await extract_memories("Hello", "Hi", "user123")
                assert result == []

    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"category": "family", "content": "Has a daughter named Sarah", "importance": 4},
        ]))]

        with patch("server.memory.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "test-key"
            with patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
                mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
                result = await extract_memories(
                    "My daughter Sarah visited today",
                    "How lovely! How is Sarah?",
                    "user123",
                )
                assert len(result) == 1
                assert result[0]["category"] == "family"
                assert "Sarah" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_filters_invalid_categories(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"category": "invalid_cat", "content": "Something", "importance": 3},
            {"category": "family", "content": "Valid fact", "importance": 3},
        ]))]

        with patch("server.memory.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "test-key"
            with patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
                mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
                result = await extract_memories("Test", "Test", "user123")
                assert len(result) == 1
                assert result[0]["category"] == "family"

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n[{"category": "health", "content": "Takes aspirin", "importance": 3}]\n```')]

        with patch("server.memory.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "test-key"
            with patch("server.memory.anthropic.AsyncAnthropic") as mock_client:
                mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
                result = await extract_memories("I take aspirin daily", "Noted", "user123")
                assert len(result) == 1
                assert result[0]["content"] == "Takes aspirin"


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_empty_memories_returns_zero(self):
        result = await deduplicate_and_store([], "user123")
        assert result == 0

    @pytest.mark.asyncio
    async def test_stores_new_memory(self):
        mock_select = MagicMock()
        mock_select.data = []  # No existing match
        mock_insert = MagicMock()
        mock_insert.data = [{"id": "new-id"}]

        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_select
        mock_table.insert.return_value.execute.return_value = mock_insert

        with patch("server.memory.get_service_client") as mock_db:
            mock_db.return_value.table.return_value = mock_table
            memories = [{"category": "family", "content": "Has a dog named Max", "importance": 3}]
            result = await deduplicate_and_store(memories, "user123")
            assert result == 1

    @pytest.mark.asyncio
    async def test_skips_duplicate(self):
        mock_select = MagicMock()
        mock_select.data = [{"id": "existing-id", "content": "Has a dog named Max", "category": "family"}]

        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_select

        with patch("server.memory.get_service_client") as mock_db:
            mock_db.return_value.table.return_value = mock_table
            memories = [{"category": "family", "content": "Has a dog named Max", "importance": 3}]
            result = await deduplicate_and_store(memories, "user123")
            assert result == 0


class TestProcessTurnMemories:
    @pytest.mark.asyncio
    async def test_handles_full_pipeline_error(self):
        with patch("server.memory.extract_memories", new_callable=AsyncMock, side_effect=Exception("fail")):
            result = await process_turn_memories("Hello", "Hi", "user123")
            assert result == []


class TestGetUserMemories:
    def test_returns_list(self):
        mock_result = MagicMock()
        mock_result.data = [{"category": "family", "content": "Test"}]

        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        with patch("server.memory.get_service_client") as mock_db:
            mock_db.return_value.table.return_value = mock_table
            result = get_user_memories("user123")
            assert len(result) == 1
