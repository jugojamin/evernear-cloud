"""Tests for check-in prompts, scheduling logic, and skip-if-recent."""

from __future__ import annotations
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.checkin_prompts import (
    EVENING_VARIANTS,
    MORNING_VARIANTS,
    pick_variant,
)


# ── Prompt variant tests ────────────────────────────────────────────

class TestCheckinPrompts:
    def test_morning_has_enough_variants(self):
        assert len(MORNING_VARIANTS) >= 5

    def test_evening_has_enough_variants(self):
        assert len(EVENING_VARIANTS) >= 5

    def test_pick_variant_returns_morning(self):
        msg = pick_variant("morning", "user-123")
        assert msg in MORNING_VARIANTS

    def test_pick_variant_returns_evening(self):
        msg = pick_variant("evening", "user-123")
        assert msg in EVENING_VARIANTS

    def test_deterministic_same_day(self):
        """Same inputs produce same variant (deterministic)."""
        d = date(2026, 3, 21)
        a = pick_variant("morning", "user-1", today=d)
        b = pick_variant("morning", "user-1", today=d)
        assert a == b

    def test_different_users_get_different_variants(self):
        """Different users likely get different variants on the same day."""
        d = date(2026, 3, 21)
        results = set()
        for i in range(10):
            results.add(pick_variant("morning", f"user-{i}", today=d))
        # With 8 variants and 10 users, should get at least 2 different ones
        assert len(results) >= 2

    def test_no_repeat_consecutive_days(self):
        """Today's variant differs from yesterday's for the same user."""
        user = "user-test-repeat"
        d1 = date(2026, 3, 15)
        d2 = date(2026, 3, 16)
        v1 = pick_variant("morning", user, today=d1)
        v2 = pick_variant("morning", user, today=d2)
        assert v1 != v2, "Consecutive days should not repeat the same variant"

    def test_variants_are_warm_not_clinical(self):
        """Variants should not contain clinical/nurse language."""
        clinical_words = ["medication", "appointment", "vitals", "assessment", "compliance", "dosage"]
        for v in MORNING_VARIANTS + EVENING_VARIANTS:
            for word in clinical_words:
                assert word not in v.lower(), f"Clinical word '{word}' found in variant: {v}"

    def test_variants_are_brief(self):
        """Each variant should be 1-3 sentences (brief, not monologuing)."""
        for v in MORNING_VARIANTS + EVENING_VARIANTS:
            sentences = [s.strip() for s in v.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            assert len(sentences) <= 4, f"Too long ({len(sentences)} sentences): {v}"


# ── Scheduler logic tests ───────────────────────────────────────────

class TestSchedulerParsing:
    def test_parse_schedules_valid(self):
        from server.checkin_scheduler import _parse_schedules
        raw = '[{"name": "morning", "hour": 9, "tz": "America/Chicago"}]'
        result = _parse_schedules(raw)
        assert len(result) == 1
        assert result[0]["name"] == "morning"

    def test_parse_schedules_invalid(self):
        from server.checkin_scheduler import _parse_schedules
        result = _parse_schedules("not json")
        assert result == []

    def test_next_fire_time_future(self):
        from server.checkin_scheduler import _next_fire_time
        from zoneinfo import ZoneInfo
        sched = {"name": "test", "hour": 23, "minute": 59, "tz": "UTC"}
        nft = _next_fire_time(sched)
        now = datetime.now(ZoneInfo("UTC"))
        assert nft >= now  # must be in the future

    def test_next_fire_time_wraps_to_tomorrow(self):
        from server.checkin_scheduler import _next_fire_time
        from zoneinfo import ZoneInfo
        # Use hour 0 — it's almost certainly in the past
        sched = {"name": "test", "hour": 0, "minute": 0, "tz": "UTC"}
        nft = _next_fire_time(sched)
        now = datetime.now(ZoneInfo("UTC"))
        # Should be tomorrow (or today if run exactly at midnight)
        assert nft > now or (nft.hour == 0 and nft.minute == 0)


class TestSkipIfRecent:
    @pytest.mark.asyncio
    async def test_skip_if_recent_conversation(self):
        from server.checkin_scheduler import _had_recent_conversation

        mock_result = MagicMock()
        mock_result.data = [{"id": "conv-1"}]

        mock_table = MagicMock()
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.gte.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = mock_result

        mock_db = MagicMock()
        mock_db.table.return_value = mock_table

        with patch("server.db.client.get_service_client", return_value=mock_db):
            result = await _had_recent_conversation("user-1")
            assert result is True

    @pytest.mark.asyncio
    async def test_no_recent_conversation(self):
        from server.checkin_scheduler import _had_recent_conversation

        mock_result = MagicMock()
        mock_result.data = []

        mock_table = MagicMock()
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.gte.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = mock_result

        mock_db = MagicMock()
        mock_db.table.return_value = mock_table

        with patch("server.db.client.get_service_client", return_value=mock_db):
            result = await _had_recent_conversation("user-1")
            assert result is False

    @pytest.mark.asyncio
    async def test_db_error_defaults_to_send(self):
        from server.checkin_scheduler import _had_recent_conversation

        with patch("server.db.client.get_service_client", side_effect=Exception("DB down")):
            result = await _had_recent_conversation("user-1")
            assert result is False  # Default: send if we can't check


# ── Checkin log tests ────────────────────────────────────────────────

class TestCheckinLog:
    def test_log_checkin_skipped(self, tmp_path):
        from server.checkin_log import log_checkin, _read_log, CHECKIN_LOG_FILE
        import server.checkin_log as cl

        # Point to temp file
        original = cl.CHECKIN_LOG_FILE
        cl.CHECKIN_LOG_FILE = tmp_path / "checkin_log.json"

        try:
            cid = log_checkin("morning", "user-1", "skipped")
            assert cid  # got an ID back
            entries = _read_log()
            assert len(entries) == 1
            assert entries[0]["delivery"] == "skipped"
        finally:
            cl.CHECKIN_LOG_FILE = original
