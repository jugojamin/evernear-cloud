"""Tests for Drift Monitoring (Phase 6)."""

import pytest
from server.metrics import TurnMetrics


class TestMetricsLogged:
    """Each of the core metrics should record correctly."""

    def test_stt_ms_recorded(self):
        m = TurnMetrics()
        m.start_stt()
        m.end_stt()
        assert m.stt_ms >= 0

    def test_llm_ttft_recorded(self):
        m = TurnMetrics()
        m.start_llm()
        m.llm_first_token_received()
        assert m.llm_ttft_ms >= 0

    def test_llm_total_recorded(self):
        m = TurnMetrics()
        m.start_llm()
        m.end_llm()
        assert m.llm_total_ms >= 0

    def test_tts_ttfb_recorded(self):
        m = TurnMetrics()
        m.start_tts()
        m.tts_first_byte_received()
        assert m.tts_ttfb_ms >= 0

    def test_total_ms_recorded(self):
        m = TurnMetrics()
        m.start_turn()
        m.end_turn()
        assert m.total_ms >= 0

    def test_model_used_recorded(self):
        m = TurnMetrics()
        m.model_used = "claude-sonnet-4-20250514"
        assert m.model_used == "claude-sonnet-4-20250514"

    def test_sonnet_reason_recorded(self):
        m = TurnMetrics()
        m.sonnet_reason = "emotional_keyword:grief"
        d = m.to_dict()
        assert d["sonnet_reason"] == "emotional_keyword:grief"

    def test_to_dict_contains_all_fields(self):
        m = TurnMetrics()
        d = m.to_dict()
        expected = {"stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ttfb_ms",
                    "total_ms", "model_used", "tts_provider", "sonnet_reason"}
        assert set(d.keys()) == expected


class TestPerUserIsolation:
    """Metrics are per-instance — no shared state between users."""

    def test_separate_instances(self):
        m1 = TurnMetrics()
        m2 = TurnMetrics()
        m1.model_used = "haiku"
        m2.model_used = "sonnet"
        assert m1.model_used != m2.model_used

    def test_no_class_level_state(self):
        m1 = TurnMetrics()
        m1.start_turn()
        m1.end_turn()
        m2 = TurnMetrics()
        assert m2.total_ms == 0
