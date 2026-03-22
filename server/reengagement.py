"""Silence/Inactivity Re-engagement — gentle check-in after extended silence."""

from __future__ import annotations
import asyncio
import logging
import random
import time
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Warm, gentle re-engagement messages — rotated randomly
REENGAGEMENT_MESSAGES = [
    "I'm still here whenever you're ready.",
    "Take your time — I'm not going anywhere.",
    "Just checking in — I'm here if you want to talk.",
    "No rush at all. I'm right here.",
    "I'm still here. Whenever you'd like to continue, I'm listening.",
]

# Inactivity threshold in seconds before re-engagement
INACTIVITY_THRESHOLD_SECONDS = 90


class InactivityMonitor:
    """Monitors user activity and triggers a single gentle re-engagement after silence.

    Usage:
        monitor = InactivityMonitor(on_reengage=send_tts_callback)
        monitor.start()
        # Call monitor.record_activity() on any user input
        # Call monitor.stop() on disconnect
    """

    def __init__(
        self,
        on_reengage: Callable[[str], Awaitable[None]],
        threshold_seconds: float = INACTIVITY_THRESHOLD_SECONDS,
        check_interval: float = 5.0,
    ):
        self._on_reengage = on_reengage
        self._threshold = threshold_seconds
        self._check_interval = check_interval
        self._last_activity: float = time.monotonic()
        self._reengagement_sent: bool = False
        self._stt_active: bool = False
        self._task: asyncio.Task | None = None
        self._stopped: bool = False

    def record_activity(self) -> None:
        """Call on any user input (audio frame, text, button press)."""
        self._last_activity = time.monotonic()
        self._reengagement_sent = False

    def set_stt_active(self, active: bool) -> None:
        """Update STT state — don't re-engage while user is speaking."""
        self._stt_active = active
        if active:
            self._last_activity = time.monotonic()

    def start(self) -> None:
        """Start the inactivity monitoring loop."""
        self._stopped = False
        self._last_activity = time.monotonic()
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        """Stop monitoring (call on disconnect)."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()

    def _pick_message(self) -> str:
        """Select a random re-engagement message."""
        return random.choice(REENGAGEMENT_MESSAGES)

    async def _monitor_loop(self) -> None:
        """Background loop checking for inactivity."""
        try:
            while not self._stopped:
                await asyncio.sleep(self._check_interval)

                if self._stopped:
                    break

                # Skip if already sent re-engagement this silence period
                if self._reengagement_sent:
                    continue

                # Skip if STT is active (user is speaking)
                if self._stt_active:
                    continue

                elapsed = time.monotonic() - self._last_activity
                if elapsed >= self._threshold:
                    message = self._pick_message()
                    logger.info(f"Inactivity re-engagement after {elapsed:.0f}s: {message}")
                    try:
                        await self._on_reengage(message)
                    except Exception as e:
                        logger.error(f"Re-engagement callback failed: {e}")
                    self._reengagement_sent = True

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"InactivityMonitor loop error: {e}")
