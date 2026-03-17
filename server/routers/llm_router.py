"""LLM Router — Haiku default, Sonnet escalation per routing rules."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-20250514"

# Emotional keywords that trigger Sonnet escalation
EMOTIONAL_KEYWORDS = {
    "grief", "loss", "scared", "lonely", "miss", "died", "worried",
    "confused", "pain", "afraid", "crying", "upset", "sad", "hurting",
    "passed away", "gone", "lost", "frightened", "anxious", "depressed",
}

# Memory categories that trigger Sonnet
SONNET_MEMORY_CATEGORIES = {"stories", "meaning", "faith", "emotions"}

# Compile pattern for efficient matching
_EMOTIONAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in EMOTIONAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    model: str
    reason: str | None = None  # None = Haiku default, string = Sonnet escalation reason


def route_llm(
    transcript: str,
    onboarding_active: bool = False,
    turn_count: int = 0,
    last_memory_categories: list[str] | None = None,
) -> RoutingDecision:
    """Determine which Claude model to use for this turn.

    Escalation rules (any match → Sonnet):
    1. Onboarding is active
    2. Emotional keywords in transcript
    3. Conversation depth > 10 turns
    4. Previous turn extracted emotion/meaning/stories/faith memories
    """
    # Rule 1: Onboarding
    if onboarding_active:
        return RoutingDecision(model=SONNET_MODEL, reason="onboarding_active")

    # Rule 2: Emotional keywords
    match = _EMOTIONAL_PATTERN.search(transcript)
    if match:
        return RoutingDecision(
            model=SONNET_MODEL,
            reason=f"emotional_keyword:{match.group(0).lower()}",
        )

    # Rule 3: Conversation depth
    if turn_count > 10:
        return RoutingDecision(model=SONNET_MODEL, reason="conversation_depth")

    # Rule 4: Previous turn memory categories
    if last_memory_categories:
        for cat in last_memory_categories:
            if cat in SONNET_MEMORY_CATEGORIES:
                return RoutingDecision(
                    model=SONNET_MODEL,
                    reason=f"memory_category:{cat}",
                )

    # Default: Haiku
    return RoutingDecision(model=HAIKU_MODEL)
