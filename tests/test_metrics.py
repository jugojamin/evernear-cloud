"""Tests for Latency Metrics tracking."""

import time
import pytest
from server.metrics import TurnMetrics


class TestTurnMetrics:
    def test_initial_values_zero(self):
        m = TurnMetrics()
        assert m.stt_ms == 0
        assert m.llm_ttft_ms == 0
        assert m.total_ms == 0

    def test_stt_timing(self):
        m = TurnMetrics()
        m.start_stt()
        time.sleep(0.01)
        m.end_stt()
        assert m.stt_ms >= 10

    def test_llm_timing(self):
        m = TurnMetrics()
        m.start_llm()
        time.sleep(0.01)
        m.llm_first_token_received()
        assert m.llm_ttft_ms >= 10
        time.sleep(0.01)
        m.end_llm()
        assert m.llm_total_ms >= 20

    def test_tts_timing(self):
        m = TurnMetrics()
        m.start_tts()
        time.sleep(0.01)
        m.tts_first_byte_received()
        assert m.tts_ttfb_ms >= 10

    def test_total_timing(self):
        m = TurnMetrics()
        m.start_turn()
        time.sleep(0.02)
        m.end_turn()
        assert m.total_ms >= 20

    def test_to_dict(self):
        m = TurnMetrics(stt_ms=150, llm_ttft_ms=300, model_used="haiku-4.5")
        d = m.to_dict()
        assert d["stt_ms"] == 150
        assert d["llm_ttft_ms"] == 300
        assert d["model_used"] == "haiku-4.5"
        assert "sonnet_reason" in d

    def test_to_dict_includes_all_fields(self):
        m = TurnMetrics()
        d = m.to_dict()
        expected_keys = {"stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ttfb_ms",
                         "total_ms", "model_used", "tts_provider", "sonnet_reason"}
        assert set(d.keys()) == expected_keys


class TestAlerts:
    def test_no_alerts_under_threshold(self):
        m = TurnMetrics(total_ms=800, stt_ms=150, llm_ttft_ms=300, tts_ttfb_ms=75)
        alerts = m.check_alerts()
        assert len(alerts) == 0

    def test_total_ms_alert(self):
        m = TurnMetrics(total_ms=2000)
        alerts = m.check_alerts()
        assert any("total_ms" in a for a in alerts)

    def test_stt_alert(self):
        m = TurnMetrics(stt_ms=500)
        alerts = m.check_alerts()
        assert any("stt_ms" in a for a in alerts)

    def test_llm_ttft_alert(self):
        m = TurnMetrics(llm_ttft_ms=800)
        alerts = m.check_alerts()
        assert any("llm_ttft_ms" in a for a in alerts)

    def test_tts_ttfb_alert(self):
        m = TurnMetrics(tts_ttfb_ms=300)
        alerts = m.check_alerts()
        assert any("tts_ttfb_ms" in a for a in alerts)

    def test_multiple_alerts(self):
        m = TurnMetrics(total_ms=2000, stt_ms=500, llm_ttft_ms=800, tts_ttfb_ms=300)
        alerts = m.check_alerts()
        assert len(alerts) == 4
