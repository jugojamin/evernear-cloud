"""Per-turn latency tracking for EverNear voice pipeline."""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Tracks timing for each stage of a conversation turn."""
    stt_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_ttfb_ms: int = 0
    context_build_ms: int = 0
    tts_total_ms: int = 0
    total_ms: int = 0
    model_used: str = "haiku-4.5"
    tts_provider: str = "cartesia"
    sonnet_reason: str | None = None

    # Internal timestamps (not serialized)
    _start_time: float = field(default=0.0, repr=False)
    _stt_start: float = field(default=0.0, repr=False)
    _llm_start: float = field(default=0.0, repr=False)
    _llm_first_token: float = field(default=0.0, repr=False)
    _tts_start: float = field(default=0.0, repr=False)
    _tts_first_byte: float = field(default=0.0, repr=False)
    _context_build_start: float = field(default=0.0, repr=False)
    _tts_end: float = field(default=0.0, repr=False)

    def start_turn(self):
        self._start_time = time.monotonic()

    def start_stt(self):
        self._stt_start = time.monotonic()

    def end_stt(self):
        if self._stt_start:
            self.stt_ms = int((time.monotonic() - self._stt_start) * 1000)

    def start_llm(self):
        self._llm_start = time.monotonic()

    def llm_first_token_received(self):
        if self._llm_start:
            self._llm_first_token = time.monotonic()
            self.llm_ttft_ms = int((self._llm_first_token - self._llm_start) * 1000)

    def end_llm(self):
        if self._llm_start:
            self.llm_total_ms = int((time.monotonic() - self._llm_start) * 1000)

    def start_context_build(self):
        self._context_build_start = time.monotonic()

    def end_context_build(self):
        if self._context_build_start:
            self.context_build_ms = int((time.monotonic() - self._context_build_start) * 1000)

    def start_tts(self):
        self._tts_start = time.monotonic()

    def tts_first_byte_received(self):
        if self._tts_start:
            self._tts_first_byte = time.monotonic()
            self.tts_ttfb_ms = int((self._tts_first_byte - self._tts_start) * 1000)

    def end_tts(self):
        if self._tts_start:
            self._tts_end = time.monotonic()
            self.tts_total_ms = int((self._tts_end - self._tts_start) * 1000)

    def end_turn(self):
        if self._start_time:
            self.total_ms = int((time.monotonic() - self._start_time) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stt_ms": self.stt_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "llm_total_ms": self.llm_total_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
            "context_build_ms": self.context_build_ms,
            "tts_total_ms": self.tts_total_ms,
            "total_ms": self.total_ms,
            "model_used": self.model_used,
            "tts_provider": self.tts_provider,
            "sonnet_reason": self.sonnet_reason,
        }

    def check_alerts(self) -> list[str]:
        """Return list of alert messages for metrics exceeding thresholds."""
        alerts = []
        if self.total_ms > 1500:
            alerts.append(f"ALERT: total_ms={self.total_ms} exceeds 1500ms threshold")
        if self.stt_ms > 400:
            alerts.append(f"ALERT: stt_ms={self.stt_ms} exceeds 400ms threshold")
        if self.llm_ttft_ms > 700:
            alerts.append(f"ALERT: llm_ttft_ms={self.llm_ttft_ms} exceeds 700ms threshold")
        if self.tts_ttfb_ms > 250:
            alerts.append(f"ALERT: tts_ttfb_ms={self.tts_ttfb_ms} exceeds 250ms threshold")
        if self.context_build_ms > 500:
            alerts.append(f"ALERT: context_build_ms={self.context_build_ms} exceeds 500ms threshold")
        for a in alerts:
            logger.warning(a)
        return alerts
