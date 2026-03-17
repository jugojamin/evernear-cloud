"""Context Builder — assembles system prompt + memories + history for each turn."""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_prompt() -> str:
    """Load the base system prompt from file."""
    path = PROMPTS_DIR / "system_prompt.txt"
    return path.read_text() if path.exists() else ""


def load_onboarding_prompt(section: str) -> str:
    """Load an onboarding section prompt."""
    path = PROMPTS_DIR / "onboarding" / f"{section}.txt"
    return path.read_text() if path.exists() else ""


def format_memory_summary(memories: list[dict[str, Any]]) -> str:
    """Compress memories into a concise summary paragraph for context injection."""
    if not memories:
        return ""

    # Group by category
    by_category: dict[str, list[str]] = {}
    for m in memories:
        cat = m.get("category", "general")
        content = m.get("content", "")
        if content:
            by_category.setdefault(cat, []).append(content)

    parts = []
    for cat, items in by_category.items():
        label = cat.replace("_", " ").title()
        summary = "; ".join(items[:5])  # Cap at 5 per category for token budget
        parts.append(f"{label}: {summary}")

    return "What you know about this person:\n" + "\n".join(parts)


def format_conversation_history(
    messages: list[dict[str, str]], max_turns: int = 8
) -> list[dict[str, str]]:
    """Format recent conversation turns for the LLM context window."""
    # Take last N messages (each turn = user + assistant = 2 messages)
    recent = messages[-(max_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def get_time_awareness() -> str:
    """Generate time-of-day context for natural conversation."""
    hour = datetime.now().hour
    if hour < 6:
        return "It's very early morning (before 6am). The person may have trouble sleeping."
    elif hour < 12:
        return "It's morning. A warm good morning greeting is appropriate if this is the first conversation today."
    elif hour < 17:
        return "It's afternoon."
    elif hour < 21:
        return "It's evening. A calmer, wind-down tone is appropriate."
    else:
        return "It's late evening/night. Be gentle and calming."


def build_context(
    user_name: str,
    memories: list[dict[str, Any]],
    conversation_history: list[dict[str, str]],
    onboarding_state: dict[str, Any] | None = None,
    max_turns: int = 8,
) -> tuple[str, list[dict[str, str]]]:
    """Build the full context for a conversation turn.

    Returns:
        (system_prompt, message_history) ready for Claude API call.
    """
    # 1. Base system prompt
    system_parts = [load_system_prompt()]

    # 2. Time awareness
    system_parts.append(get_time_awareness())

    # 3. User's preferred name
    if user_name:
        system_parts.append(f"The person you're talking to prefers to be called: {user_name}")

    # 4. Memory summary
    mem_summary = format_memory_summary(memories)
    if mem_summary:
        system_parts.append(mem_summary)

    # 5. Onboarding context
    if onboarding_state and not onboarding_state.get("completed", False):
        section = onboarding_state.get("current_section", "")
        if section:
            onb_prompt = load_onboarding_prompt(section)
            if onb_prompt:
                system_parts.append(f"ONBOARDING — Current Section: {section}\n{onb_prompt}")

    system_prompt = "\n\n".join(p for p in system_parts if p)

    # 6. Conversation history
    history = format_conversation_history(conversation_history, max_turns)

    return system_prompt, history
