"""Tests for the Pipecat pipeline — text mode processing."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from server.pipeline import EverNearPipeline


class TestPipelineInit:
    def test_creates_with_user_id(self):
        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            p = EverNearPipeline(user_id="user-123")
            assert p.user_id == "user-123"
            assert p.turn_count == 0

    def test_generates_conversation_id(self):
        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            p = EverNearPipeline(user_id="user-123")
            assert p.conversation_id is not None
            assert len(p.conversation_id) > 0

    def test_custom_conversation_id(self):
        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            p = EverNearPipeline(user_id="user-123", conversation_id="conv-456")
            assert p.conversation_id == "conv-456"


class TestConversationHistory:
    def test_history_accumulates(self):
        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            p = EverNearPipeline(user_id="user-123")
            p.conversation_history.append({"role": "user", "content": "Hello"})
            p.conversation_history.append({"role": "assistant", "content": "Hi"})
            assert len(p.conversation_history) == 2
