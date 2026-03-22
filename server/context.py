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


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we",
    "you", "your", "he", "she", "it", "they", "them", "his", "her",
    "its", "their", "what", "which", "who", "this", "that", "of", "in",
    "to", "for", "with", "on", "at", "from", "by", "about", "as",
    "and", "but", "or", "not", "so", "very", "just", "how",
})


def _content_words(text: str) -> set[str]:
    """Extract meaningful words from text (excluding stop words)."""
    words = set(text.lower().split())
    return words - _STOP_WORDS


def _score_relevance(memory_content: str, user_words: set[str]) -> int:
    """Score a memory's relevance to the current user message by word overlap."""
    if not user_words:
        return 0
    mem_words = _content_words(memory_content)
    return len(mem_words & user_words)


def format_memory_summary(
    memories: list[dict[str, Any]],
    user_message: str = "",
    token_budget: int = 800,
) -> str:
    """Compress memories into a concise summary for context injection.
    
    When user_message is provided, memories relevant to the current conversation
    are promoted; background memories get a lower cap per category.
    Token budget is approximate (~4 chars per token).
    """
    if not memories:
        return ""

    user_words = _content_words(user_message) if user_message else set()

    # Split into relevant vs background
    relevant: list[dict[str, Any]] = []
    background: list[dict[str, Any]] = []
    for m in memories:
        content = m.get("content", "")
        if not content:
            continue
        if user_words and _score_relevance(content, user_words) > 0:
            relevant.append(m)
        else:
            background.append(m)

    # Build output with token budget
    char_budget = token_budget * 4  # ~4 chars per token
    parts = []
    chars_used = 0

    # Relevant memories first — include up to 5 per category
    if relevant:
        by_cat: dict[str, list[str]] = {}
        for m in relevant:
            cat = m.get("category", "general")
            by_cat.setdefault(cat, []).append(m["content"])
        for cat, items in by_cat.items():
            label = cat.replace("_", " ").title()
            summary = "; ".join(items[:5])
            line = f"You know that — {label}: {summary}"
            if chars_used + len(line) > char_budget:
                break
            parts.append(line)
            chars_used += len(line)

    # Background memories — cap at 2 per category
    bg_cap = 2 if user_words else 5  # If no user message, use original behavior
    by_cat_bg: dict[str, list[str]] = {}
    for m in background:
        cat = m.get("category", "general")
        by_cat_bg.setdefault(cat, []).append(m["content"])
    for cat, items in by_cat_bg.items():
        label = cat.replace("_", " ").title()
        summary = "; ".join(items[:bg_cap])
        line = f"{label}: {summary}"
        if chars_used + len(line) > char_budget:
            logger.warning(f"Memory context trimmed: budget exceeded at {chars_used} chars")
            break
        parts.append(line)
        chars_used += len(line)

    if not parts:
        return ""
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
    user_message: str = "",
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

    # 4. Memory summary (relevance-aware)
    mem_summary = format_memory_summary(memories, user_message=user_message)
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

    # 6. Token budget check — if system prompt > 2000 tokens (~8000 chars), trim
    max_system_chars = 8000  # ~2000 tokens
    if len(system_prompt) > max_system_chars:
        logger.warning(f"System prompt too large ({len(system_prompt)} chars), regenerating with tighter memory budget")
        # Regenerate with reduced memory budget
        trimmed_summary = format_memory_summary(memories, user_message=user_message, token_budget=400)
        system_parts_trimmed = [p for p in system_parts if "What you know" not in p]
        if trimmed_summary:
            system_parts_trimmed.append(trimmed_summary)
        system_prompt = "\n\n".join(p for p in system_parts_trimmed if p)

    # 7. Conversation history
    history = format_conversation_history(conversation_history, max_turns)

    return system_prompt, history
