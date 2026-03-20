"""Proactive check-in scheduler — fires check-in events at configured times."""

from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from server.checkin_log import log_checkin
from server.config import get_settings
from server.incident_log import log_incident
from server.notifications import push_service
from server.transport import manager

logger = logging.getLogger(__name__)

# Placeholder message — will be replaced when prompt design is done
CHECKIN_PLACEHOLDER = "Hi there — just checking in."


def _parse_schedules(raw: str) -> list[dict]:
    """Parse JSON schedule string from config."""
    try:
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to parse checkin_schedules: {e}")
        return []


def _next_fire_time(schedule: dict) -> datetime:
    """Calculate the next fire time for a schedule entry."""
    tz = ZoneInfo(schedule.get("tz", "UTC"))
    now = datetime.now(tz)
    target = now.replace(
        hour=schedule["hour"],
        minute=schedule.get("minute", 0),
        second=0,
        microsecond=0,
    )
    if target <= now:
        target += timedelta(days=1)
    return target


async def _fire_checkin(schedule_name: str):
    """Execute a single check-in event for all connected or known users."""
    from server.db.client import get_service_client

    settings = get_settings()

    # Get all users (single-user system for now, but future-safe)
    try:
        db = get_service_client()
        users = (db.table("users").select("id").execute()).data or []
    except Exception as e:
        logger.error(f"Check-in failed to fetch users: {e}")
        log_incident("transport", "checkin_scheduler.py", f"Check-in user fetch failed: {e}", fallback_triggered=False)
        return

    for user in users:
        user_id = user["id"]

        # Try WebSocket first
        if manager.is_connected(user_id):
            try:
                from uuid import uuid4
                checkin_id = log_checkin(schedule_name, user_id, "websocket")
                await manager.send_json(user_id, {
                    "type": "checkin",
                    "checkin_id": checkin_id,
                    "schedule_name": schedule_name,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": CHECKIN_PLACEHOLDER,
                })
                logger.info(f"Check-in fired: schedule={schedule_name}, user={user_id}, delivery=websocket, checkin_id={checkin_id}")
                continue
            except Exception as e:
                logger.error(f"Check-in WebSocket delivery failed for {user_id}: {e}")

        # Fall back to push notification
        try:
            db = get_service_client()
            profile = (db.table("users").select("device_token").eq("id", user_id).single().execute()).data
            device_token = (profile or {}).get("device_token")

            if device_token:
                success = await push_service.send_notification(
                    device_token=device_token,
                    title="EverNear",
                    body=CHECKIN_PLACEHOLDER,
                    category="checkin",
                )
                delivery = "push" if success else "failed"
            else:
                delivery = "failed"
                logger.warning(f"No device token for user {user_id} — check-in not delivered")

            checkin_id = log_checkin(schedule_name, user_id, delivery)
            logger.info(f"Check-in fired: schedule={schedule_name}, user={user_id}, delivery={delivery}, checkin_id={checkin_id}")

            if delivery == "failed":
                log_incident("transport", "checkin_scheduler.py", f"Check-in delivery failed for {user_id}", fallback_triggered=False)

        except Exception as e:
            logger.error(f"Check-in push delivery failed for {user_id}: {e}")
            log_checkin(schedule_name, user_id, "failed")
            log_incident("transport", "checkin_scheduler.py", f"Check-in push failed: {e}", fallback_triggered=False)


async def run_scheduler():
    """Main scheduler loop — runs as background task during server lifespan."""
    settings = get_settings()

    if not settings.checkin_enabled:
        logger.info("Check-in scheduler disabled (checkin_enabled=false)")
        return

    schedules = _parse_schedules(settings.checkin_schedules)
    if not schedules:
        logger.warning("Check-in scheduler enabled but no schedules configured")
        return

    logger.info(f"Check-in scheduler started with {len(schedules)} schedule(s)")

    while True:
        # Find the next fire time across all schedules
        next_fires = []
        for sched in schedules:
            nft = _next_fire_time(sched)
            next_fires.append((nft, sched))

        next_fires.sort(key=lambda x: x[0])
        next_time, next_sched = next_fires[0]

        # Sleep until next fire time
        now = datetime.now(ZoneInfo(next_sched.get("tz", "UTC")))
        sleep_seconds = max(0, (next_time - now).total_seconds())
        logger.info(f"Next check-in '{next_sched['name']}' in {sleep_seconds:.0f}s at {next_time.isoformat()}")

        await asyncio.sleep(sleep_seconds)

        # Fire the check-in
        try:
            await _fire_checkin(next_sched["name"])
        except Exception as e:
            logger.error(f"Check-in scheduler error: {e}")
            log_incident("transport", "checkin_scheduler.py", f"Scheduler error: {e}", fallback_triggered=False)
