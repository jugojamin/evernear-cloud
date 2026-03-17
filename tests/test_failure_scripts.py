"""Tests for Failure Scripts (Phase 5)."""

import pytest
from server.failure_scripts import (
    FAILURE_SCRIPTS,
    REPEATED_FAILURE_THRESHOLD,
    get_failure_response,
)


class TestEachFailureType:
    """Each failure type should have a correct script."""

    @pytest.mark.parametrize("failure_type", [
        "api_timeout",
        "stt_failure",
        "llm_error",
        "websocket_disconnect",
        "repeated_failure",
        "complete_outage",
        "validator_failure",
        "validator_timeout",
        "memory_cache_failure",
    ])
    def test_failure_type_has_response(self, failure_type):
        response = get_failure_response(failure_type)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_tts_failure_returns_none(self):
        """TTS failure falls back to text — returns None."""
        assert FAILURE_SCRIPTS["tts_failure"] is None


class TestRepeatedFailure:
    def test_switches_to_repeated_after_threshold(self):
        response = get_failure_response("api_timeout", session_failure_count=3)
        assert response == FAILURE_SCRIPTS["repeated_failure"]

    def test_normal_below_threshold(self):
        response = get_failure_response("api_timeout", session_failure_count=1)
        assert response == FAILURE_SCRIPTS["api_timeout"]

    def test_threshold_value(self):
        assert REPEATED_FAILURE_THRESHOLD == 3


class TestNoTechnicalDetails:
    """No failure script should contain technical language."""

    TECHNICAL_TERMS = [
        "error", "exception", "traceback", "stack", "timeout",
        "500", "404", "null", "undefined", "NoneType",
        "API", "HTTP", "WebSocket", "SSL", "TCP",
    ]

    @pytest.mark.parametrize("failure_type,script", [
        (k, v) for k, v in FAILURE_SCRIPTS.items() if v is not None
    ])
    def test_no_technical_language(self, failure_type, script):
        script_lower = script.lower()
        for term in self.TECHNICAL_TERMS:
            assert term.lower() not in script_lower, (
                f"Failure script '{failure_type}' contains technical term '{term}': {script}"
            )
