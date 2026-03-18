"""Shared incident logger for all failure sites.

Writes to a persistent JSON file. Keeps last 50 entries.
NOTE: /data volume not mounted on Fly.io — using /app/incidents.json.
This survives deploys but not machine replacement. Creating a Fly volume
is an Operator decision (documented in INBOX.md).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

# Default: /app/incidents.json (in-container working dir).
# Override via INCIDENTS_FILE env var. /data/incidents.json if volume exists.
_DEFAULT_PATH = "/app/incidents.json"
if os.path.isdir("/data"):
    _DEFAULT_PATH = "/data/incidents.json"

INCIDENTS_FILE = Path(os.environ.get("INCIDENTS_FILE", _DEFAULT_PATH))
MAX_ENTRIES = 50


def log_incident(
    error_type: str,
    location: str,
    short_message: str,
    fallback_triggered: bool,
    request_id: str | None = None,
) -> str | None:
    """Append an incident entry. Returns the incident_id or None on failure."""
    incident_id = f"INC-{uuid4().hex[:8]}"
    entry = {
        "incident_id": incident_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "error_type": error_type,
        "location": location,
        "short_message": short_message,
        "request_id": request_id,
        "fallback_triggered": fallback_triggered,
    }

    try:
        entries: list[dict] = []
        if INCIDENTS_FILE.exists():
            try:
                entries = json.loads(INCIDENTS_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append(entry)
        entries = entries[-MAX_ENTRIES:]
        INCIDENTS_FILE.write_text(json.dumps(entries, indent=2))
        return incident_id
    except Exception as e:
        logger.error(f"Failed to write incident log: {e}")
        return None


def read_incidents() -> list[dict]:
    """Read all logged incidents."""
    try:
        if INCIDENTS_FILE.exists():
            return json.loads(INCIDENTS_FILE.read_text())
    except (json.JSONDecodeError, OSError, Exception):
        pass
    return []
