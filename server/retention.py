"""Data retention — daily cleanup of expired records per configurable windows."""

from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from server.config import get_settings
from server.db.client import get_service_client

logger = logging.getLogger(__name__)

_RETENTION_STATE_FILE = Path("/app/retention_state.json")


def _read_state() -> dict:
    try:
        if _RETENTION_STATE_FILE.exists():
            return json.loads(_RETENTION_STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def _write_state(state: dict):
    try:
        _RETENTION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _RETENTION_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.error(f"Failed to write retention state: {e}")


async def run_cleanup():
    """Execute retention cleanup for all configured tables."""
    settings = get_settings()
    db = get_service_client()
    now = datetime.now(timezone.utc)
    results = {}

    # Table configs: (table_name, date_column, retention_days_setting)
    table_configs = [
        ("messages", "created_at", settings.retention_messages_days),
        ("conversations", "started_at", settings.retention_conversations_days),
        ("memories", "created_at", settings.retention_memories_days),
        ("consent_logs", "created_at", 365),
    ]

    for table, date_col, retention_days in table_configs:
        if retention_days <= 0:
            results[table] = {"skipped": True, "reason": "retention=0 (never expire)"}
            continue

        cutoff = (now - timedelta(days=retention_days)).isoformat()
        try:
            # For messages, need to handle FK: delete messages first via conversation join
            if table == "messages":
                resp = db.table(table).delete().lt(date_col, cutoff).execute()
            else:
                resp = db.table(table).delete().lt(date_col, cutoff).execute()

            count = len(resp.data) if resp.data else 0
            results[table] = {"deleted": count, "cutoff": cutoff}
            if count > 0:
                logger.info(f"Retention cleanup: {table} — {count} rows older than {retention_days}d removed")
            else:
                logger.info(f"Retention cleanup: {table} — nothing to remove (cutoff: {cutoff})")
        except Exception as e:
            logger.error(f"Retention cleanup failed for {table}: {e}")
            results[table] = {"error": str(e)}

    # JSON file cleanup (incidents, checkins, diagnostics)
    for name, file_path, days in [
        ("incidents", Path("/app/incidents.json"), settings.retention_incidents_days),
        ("checkins", Path("/app/checkin_log.json"), settings.retention_checkins_days),
        ("diagnostics", Path("/app/diagnostics.json"), settings.retention_diagnostics_days),
    ]:
        if days <= 0:
            results[name] = {"skipped": True, "reason": "retention=0"}
            continue
        try:
            if file_path.exists():
                entries = json.loads(file_path.read_text())
                cutoff = (now - timedelta(days=days)).isoformat()
                before = len(entries)
                entries = [e for e in entries if e.get("ts", e.get("received_at", "")) >= cutoff]
                removed = before - len(entries)
                file_path.write_text(json.dumps(entries, indent=2))
                results[name] = {"deleted": removed, "remaining": len(entries)}
                if removed > 0:
                    logger.info(f"Retention cleanup: {name} — {removed} entries older than {days}d removed")
            else:
                results[name] = {"deleted": 0, "reason": "file not found"}
        except Exception as e:
            logger.error(f"Retention cleanup failed for {name}: {e}")
            results[name] = {"error": str(e)}

    # Save state
    state = {
        "last_run": now.isoformat(),
        "results": results,
    }
    _write_state(state)
    logger.info(f"Retention cleanup complete: {json.dumps(results)}")
    return state


def get_retention_status() -> dict:
    """Return current retention config + last run state."""
    settings = get_settings()
    state = _read_state()
    return {
        "config": {
            "messages_days": settings.retention_messages_days,
            "conversations_days": settings.retention_conversations_days,
            "memories_days": settings.retention_memories_days,
            "incidents_days": settings.retention_incidents_days,
            "checkins_days": settings.retention_checkins_days,
            "diagnostics_days": settings.retention_diagnostics_days,
        },
        "last_run": state.get("last_run"),
        "results": state.get("results", {}),
    }


async def run_scheduler():
    """Background task — runs cleanup once daily at 3am CT."""
    tz = ZoneInfo("America/Chicago")
    while True:
        now = datetime.now(tz)
        target = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        sleep_s = (target - now).total_seconds()
        logger.info(f"Retention scheduler: next cleanup in {sleep_s:.0f}s at {target.isoformat()}")
        await asyncio.sleep(sleep_s)

        try:
            await run_cleanup()
        except Exception as e:
            logger.error(f"Retention scheduler error: {e}")
