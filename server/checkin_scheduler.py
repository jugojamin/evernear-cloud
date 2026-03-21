"""Proactive check-in scheduler — fires check-in events at configured times."""

from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from server.checkin_log import log_checkin
from server.checkin_prompts import pick_variant
from server.config import get_settings
from server.incident_log import log_incident
from server.notifications import push_service
from server.transport import manager

logger = logging.getLogger(__name__)

# How recently must the user have had a conversation to skip check-in
SKIP_IF_RECENT_HOURS = 2


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


async def _had_recent_conversation(user_id: str, hours: int = SKIP_IF_RECENT_HOURS) -> bool:
    """Check if user had a conversation within the last N hours."""
    try:
        from server.db.client import get_service_client
        db = get_service_client()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        result = (
            db.table("conversations")
            .select("id")
            .eq("user_id", user_id)
            .gte("started_at", cutoff)
            .limit(1)
            .execute()
        )
        had_recent = bool(result.data)
        if had_recent:
            logger.info(f"User {user_id} had recent conversation — skipping check-in")
        return had_recent
    except Exception as e:
        logger.warning(f"Could not check recent conversations for {user_id}: {e}")
        return False  # Default to sending if we can't check


def _get_user_timezone(user: dict) -> str:
    """Get user's timezone, falling back to schedule tz or UTC."""
    return user.get("timezone") or "UTC"


async def _fire_checkin(schedule_name: str):
    """Execute a single check-in event for all known users."""
    from server.db.client import get_service_client

    settings = get_settings()

    try:
        db = get_service_client()
        users = (db.table("users").select("id, device_token, timezone, preferred_name").execute()).data or []
    except Exception as e:
        logger.error(f"Check-in failed to fetch users: {e}")
        log_incident("transport", "checkin_scheduler.py", f"Check-in user fetch failed: {e}", fallback_triggered=False)
        return

    for user in users:
        user_id = user["id"]

        # Skip if user had a recent conversation
        if await _had_recent_conversation(user_id):
            log_checkin(schedule_name, user_id, "skipped")
            continue

        # Pick a varied, natural prompt for this user + day
        message = pick_variant(schedule_name, user_id)

        # Personalize with name if available
        preferred_name = user.get("preferred_name")
        if preferred_name and not message.lower().startswith(f"hi {preferred_name.lower()}"):
            # Some variants start with "Hi there" or "Hey" — personalize them
            for generic in ["Hi there", "Hey there"]:
                if message.startswith(generic):
                    message = message.replace(generic, f"Hi {preferred_name}", 1)
                    break

        # Try WebSocket first
        if manager.is_connected(user_id):
            try:
                checkin_id = log_checkin(schedule_name, user_id, "websocket")
                await manager.send_json(user_id, {
                    "type": "checkin",
                    "checkin_id": checkin_id,
                    "schedule_name": schedule_name,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": message,
                })
                logger.info(f"Check-in fired: schedule={schedule_name}, user={user_id}, delivery=websocket")
                continue
            except Exception as e:
                logger.error(f"Check-in WebSocket delivery failed for {user_id}: {e}")

        # Fall back to push notification
        device_token = user.get("device_token")
        if device_token:
            try:
                success = await push_service.send_notification(
                    device_token=device_token,
                    title="EverNear",
                    body=message,
                    category="checkin",
                )
                delivery = "push" if success else "failed"
            except Exception as e:
                logger.error(f"Check-in push delivery failed for {user_id}: {e}")
                delivery = "failed"
        else:
            delivery = "no_token"
            logger.warning(f"No device token for user {user_id} — check-in not delivered")

        checkin_id = log_checkin(schedule_name, user_id, delivery)
        logger.info(f"Check-in fired: schedule={schedule_name}, user={user_id}, delivery={delivery}")

        if delivery in ("failed", "no_token"):
            log_incident("transport", "checkin_scheduler.py", f"Check-in delivery={delivery} for {user_id}", fallback_triggered=False)


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
        next_fires = []
        for sched in schedules:
            nft = _next_fire_time(sched)
            next_fires.append((nft, sched))

        next_fires.sort(key=lambda x: x[0])
        next_time, next_sched = next_fires[0]

        now = datetime.now(ZoneInfo(next_sched.get("tz", "UTC")))
        sleep_seconds = max(0, (next_time - now).total_seconds())
        logger.info(f"Next check-in '{next_sched['name']}' in {sleep_seconds:.0f}s at {next_time.isoformat()}")

        await asyncio.sleep(sleep_seconds)

        try:
            await _fire_checkin(next_sched["name"])
        except Exception as e:
            logger.error(f"Check-in scheduler error: {e}")
            log_incident("transport", "checkin_scheduler.py", f"Scheduler error: {e}", fallback_triggered=False)
