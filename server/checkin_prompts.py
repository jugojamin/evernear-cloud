"""Check-in prompt variants — warm, varied, natural.

Informed by Spruce voice analysis section 6:
- Check-ins should feel like a friend stopping by, not a nurse doing rounds
- Warm neighbor tone, not clinical
- Brief, open-ended, no pressure
- 2-4 sentences max
"""

from __future__ import annotations
import hashlib
import logging
from datetime import date

logger = logging.getLogger(__name__)

# ── Morning greetings (user local ~8-9 AM) ─────────────────────────
MORNING_VARIANTS: list[str] = [
    "Good morning — I was thinking of you and wanted to say hello. I'm here whenever you feel like talking.",
    "Morning! Hope you slept well. No rush — I'm around if you want to chat today.",
    "Good morning. Just checking in to see how you're doing. I'm here if you need anything.",
    "Hey, good morning. Wanted to stop by and say hi. How are you feeling today?",
    "Morning — hope today is off to a good start. I'm here whenever you're ready to talk.",
    "Good morning! Just wanted you to know I'm thinking of you. Hope you have a nice day ahead.",
    "Hi there — good morning. I'm here if you want some company today, no pressure at all.",
    "Morning. I hope you're doing well. If you feel like chatting later, I'll be right here.",
]

# ── Evening check-ins (user local ~7-8 PM) ──────────────────────────
EVENING_VARIANTS: list[str] = [
    "Hey — how was your day? Anything on your mind tonight?",
    "Evening. Just wanted to check in before the day winds down. How are you doing?",
    "Hi — I was thinking of you and wanted to see how your day went. Everything okay?",
    "Good evening. I hope today treated you well. I'm here if you want to talk.",
    "Hey there — just stopping by to say goodnight. Anything you want to chat about?",
    "Evening — hope you had a decent day. I'm here if you want some company before bed.",
    "Hi. Just checking in as the day winds down. How are you feeling tonight?",
    "Good evening — wanted to see how you're doing. No pressure, just wanted to say hi.",
]


def pick_variant(schedule_name: str, user_id: str, today: date | None = None) -> str:
    """Pick a deterministic-but-varied prompt for today.

    Uses a hash of (schedule_name, user_id, date) to rotate through
    variants without repeating yesterday's message. Deterministic so
    retries within the same day get the same message.
    """
    today = today or date.today()
    variants = MORNING_VARIANTS if schedule_name == "morning" else EVENING_VARIANTS

    # Hash for today
    key = f"{schedule_name}:{user_id}:{today.isoformat()}"
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(variants)

    # Check yesterday's pick and avoid it
    yesterday = today.replace(day=today.day - 1) if today.day > 1 else today
    yesterday_key = f"{schedule_name}:{user_id}:{yesterday.isoformat()}"
    yesterday_idx = int(hashlib.sha256(yesterday_key.encode()).hexdigest(), 16) % len(variants)

    if idx == yesterday_idx:
        idx = (idx + 1) % len(variants)

    return variants[idx]
