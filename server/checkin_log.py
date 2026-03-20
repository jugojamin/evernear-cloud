"""Check-in event persistence — tracks proactive check-in delivery and response rates."""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

CHECKIN_LOG_FILE = Path("/app/checkin_log.json")
MAX_ENTRIES = 100


def _read_log() -> list[dict]:
    try:
        if CHECKIN_LOG_FILE.exists():
            return json.loads(CHECKIN_LOG_FILE.read_text())
    except Exception as e:
        logger.error(f"Failed to read checkin log: {e}")
    return []


def _write_log(entries: list[dict]):
    try:
        CHECKIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKIN_LOG_FILE.write_text(json.dumps(entries[-MAX_ENTRIES:], indent=2))
    except Exception as e:
        logger.error(f"Failed to write checkin log: {e}")


def log_checkin(
    schedule_name: str,
    user_id: str,
    delivery: str,  # "websocket" | "push" | "failed"
) -> str:
    """Record a check-in event. Returns checkin_id."""
    checkin_id = str(uuid4())[:8]
    entry = {
        "checkin_id": checkin_id,
        "schedule_name": schedule_name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "delivery": delivery,
        "responded": False,
        "response_ts": None,
    }
    entries = _read_log()
    entries.append(entry)
    _write_log(entries)
    return checkin_id


def mark_responded(user_id: str):
    """Mark the most recent unresponded check-in for this user as responded."""
    entries = _read_log()
    for entry in reversed(entries):
        if entry["user_id"] == user_id and not entry["responded"]:
            entry["responded"] = True
            entry["response_ts"] = datetime.now(timezone.utc).isoformat()
            _write_log(entries)
            return
    # No unresponded check-in found — nothing to mark


def get_checkins(limit: int = 20) -> tuple[list[dict], float]:
    """Return recent check-ins and response rate."""
    entries = _read_log()
    recent = list(reversed(entries[-limit:]))
    total = len(entries)
    responded = sum(1 for e in entries if e.get("responded"))
    rate = round(responded / total, 2) if total else 0.0
    return recent, rate
