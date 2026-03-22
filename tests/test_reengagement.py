"""Tests for Silence/Inactivity Re-engagement."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch

from server.reengagement import (
    InactivityMonitor,
    REENGAGEMENT_MESSAGES,
    INACTIVITY_THRESHOLD_SECONDS,
)


class TestInactivityMonitor:
    @pytest.mark.asyncio
    async def test_fires_after_threshold(self):
        """Re-engagement fires after inactivity threshold."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.1, check_interval=0.05)
        monitor.start()
        try:
            await asyncio.sleep(0.3)
            callback.assert_called_once()
            msg = callback.call_args[0][0]
            assert msg in REENGAGEMENT_MESSAGES
        finally:
            monitor.stop()

    @pytest.mark.asyncio
    async def test_resets_on_activity(self):
        """Timer resets when user activity is recorded."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.2, check_interval=0.05)
        monitor.start()
        try:
            await asyncio.sleep(0.1)
            monitor.record_activity()  # Reset timer
            await asyncio.sleep(0.1)
            monitor.record_activity()  # Reset again
            await asyncio.sleep(0.1)
            # Should NOT have fired — activity kept resetting
            callback.assert_not_called()
        finally:
            monitor.stop()

    @pytest.mark.asyncio
    async def test_only_one_reengagement_per_silence(self):
        """Only one re-engagement message per silence period."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.1, check_interval=0.05)
        monitor.start()
        try:
            await asyncio.sleep(0.5)  # Wait well past threshold
            # Should have been called exactly once, not multiple times
            assert callback.call_count == 1
        finally:
            monitor.stop()

    @pytest.mark.asyncio
    async def test_no_fire_during_stt(self):
        """Re-engagement does not fire while STT is active."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.1, check_interval=0.05)
        monitor.set_stt_active(True)
        monitor.start()
        try:
            await asyncio.sleep(0.3)
            callback.assert_not_called()
        finally:
            monitor.stop()

    @pytest.mark.asyncio
    async def test_fires_after_stt_ends(self):
        """Re-engagement fires after STT becomes inactive and threshold passes."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.1, check_interval=0.05)
        monitor.set_stt_active(True)
        monitor.start()
        try:
            await asyncio.sleep(0.15)
            callback.assert_not_called()
            monitor.set_stt_active(False)
            await asyncio.sleep(0.25)
            callback.assert_called_once()
        finally:
            monitor.stop()

    @pytest.mark.asyncio
    async def test_activity_after_reengagement_allows_new_one(self):
        """After re-engagement sent, new activity resets and allows another."""
        callback = AsyncMock()
        monitor = InactivityMonitor(on_reengage=callback, threshold_seconds=0.1, check_interval=0.05)
        monitor.start()
        try:
            await asyncio.sleep(0.25)
            assert callback.call_count == 1
            monitor.record_activity()  # User responded
            await asyncio.sleep(0.25)  # Go silent again
            assert callback.call_count == 2
        finally:
            monitor.stop()


class TestReengagementMessages:
    def test_sufficient_variants(self):
        """At least 3 message variants exist."""
        assert len(REENGAGEMENT_MESSAGES) >= 3

    def test_messages_are_nonempty(self):
        for msg in REENGAGEMENT_MESSAGES:
            assert len(msg) > 10

    def test_default_threshold(self):
        assert INACTIVITY_THRESHOLD_SECONDS == 90
