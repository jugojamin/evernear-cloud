"""Memory Extractor — async post-turn fact extraction and deduplication."""

from __future__ import annotations
import asyncio
import json
import logging
import re
import string
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
from uuid import UUID

import anthropic

from server.config import get_settings
from server.db.client import get_service_client

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_PATH = Path(__file__).parent / "prompts" / "memory_extraction.txt"

MEMORY_CATEGORIES = [
    "family", "health", "preferences", "stories", "emotions",
    "meaning", "culture", "faith", "interests", "caregivers", "routine",
]

# Correction patterns — signals the user is correcting a previous fact
CORRECTION_PATTERNS = [
    r"\bactually\b", r"\bno,?\s+it'?s\b", r"\bi meant\b",
    r"\bnot\s+\w+,?\s+it'?s\b", r"\bcorrection\b",
    r"\bthat'?s\s+(?:not\s+)?(?:right|wrong|incorrect)\b",
    r"\bi\s+(?:was|am)\s+wrong\b",
]

# Stop words excluded from relevance scoring
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "his", "her", "its", "their", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "very", "just", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too",
})


def _load_extraction_prompt() -> str:
    if EXTRACTION_PROMPT_PATH.exists():
        return EXTRACTION_PROMPT_PATH.read_text()
    return ""


async def extract_memories(
    user_message: str,
    assistant_message: str,
    user_id: str,
    source_turn_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structured facts from a conversation turn using Claude Haiku.

    Runs asynchronously — does NOT block the response pipeline.
    Returns list of extracted memory dicts.
    """
    s = get_settings()
    if not s.anthropic_api_key:
        logger.warning("No Anthropic API key — skipping memory extraction")
        return []

    extraction_prompt = _load_extraction_prompt()

    prompt = f"""{extraction_prompt}

Conversation turn:
User: {user_message}
Assistant: {assistant_message}

Extract any new facts worth remembering. Return a JSON array of objects with keys:
- "category": one of {MEMORY_CATEGORIES}
- "content": the fact to remember (concise, third person)
- "importance": 1-5 (5 = very important)
- "confidence": 0.0-1.0 (1.0 = explicitly stated by the user, 0.5 = implied/contextual, 0.0 = ambiguous/inferred)

If nothing worth remembering, return an empty array: []
Return ONLY valid JSON, no other text."""

    try:
        client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        memories = json.loads(text)
        if not isinstance(memories, list):
            return []

        # Validate each memory
        valid = []
        for m in memories:
            if (
                isinstance(m, dict)
                and m.get("category") in MEMORY_CATEGORIES
                and m.get("content")
                and isinstance(m.get("importance", 3), int)
            ):
                # Ensure confidence is a valid float
                conf = m.get("confidence", 0.5)
                if not isinstance(conf, (int, float)):
                    conf = 0.5
                m["confidence"] = max(0.0, min(1.0, float(conf)))
                valid.append(m)

        return valid

    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        return []


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison: lowercase, strip punctuation, remove articles."""
    text = text.lower()
    # Handle possessives before stripping punctuation
    text = re.sub(r"'s\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove common filler words
    text = re.sub(r"\b(a|an|the|is|are|was|my|his|her|their|our|name|named|called)\b", "", text)
    return " ".join(text.split())


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two strings are near-duplicates.
    
    Returns True if one is a substring of the other OR word overlap > 70%.
    """
    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)

    if not norm_a or not norm_b:
        return False

    # Substring check
    if norm_a in norm_b or norm_b in norm_a:
        return True

    # Word overlap check
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    if not words_a or not words_b:
        return False
    smaller = min(len(words_a), len(words_b))
    overlap = len(words_a & words_b)
    return (overlap / smaller) > 0.7


def _is_correction(user_message: str) -> bool:
    """Detect if the user message contains a correction pattern."""
    msg_lower = user_message.lower()
    return any(re.search(pat, msg_lower) for pat in CORRECTION_PATTERNS)


def _recency_score(created_at: str | None, now: datetime | None = None) -> float:
    """Calculate recency factor: 1.0 for today, decaying to 0.3 floor over 30 days."""
    if not created_at:
        return 0.3  # No timestamp → treat as old
    try:
        if now is None:
            now = datetime.now(timezone.utc)
        # Parse ISO timestamp
        if isinstance(created_at, str):
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            ts = created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        days_ago = (now - ts).total_seconds() / 86400
        if days_ago <= 0:
            return 1.0
        if days_ago >= 30:
            return 0.3
        # Linear decay from 1.0 to 0.3 over 30 days
        return 1.0 - (days_ago / 30) * 0.7
    except (ValueError, TypeError):
        return 0.3


async def deduplicate_and_store(
    memories: list[dict[str, Any]],
    user_id: str,
    source_turn_id: str | None = None,
    user_message: str | None = None,
) -> int:
    """Deduplicate against existing memories and store new ones. Returns count stored.
    
    Uses fuzzy matching for deduplication and correction-aware supersede logic.
    """
    if not memories:
        return 0

    db = get_service_client()
    stored = 0
    is_correction = _is_correction(user_message) if user_message else False

    for mem in memories:
        # Fetch all active memories in same category for fuzzy matching
        existing_all = (
            db.table("memories")
            .select("*")
            .eq("user_id", user_id)
            .eq("category", mem["category"])
            .eq("active", True)
            .execute()
        )

        # Check for fuzzy duplicates
        found_match = False
        for existing_mem in (existing_all.data or []):
            if _fuzzy_match(existing_mem["content"], mem["content"]):
                if is_correction:
                    # Supersede the old memory
                    db.table("memories").update({"active": False}).eq("id", existing_mem["id"]).execute()
                    logger.info(f"Superseded memory: [{existing_mem['content'][:50]}] → [{mem['content'][:50]}]")
                else:
                    # Plain duplicate — skip
                    logger.debug(f"Duplicate memory skipped: {mem['content'][:50]}")
                    found_match = True
                    break

        if found_match:
            continue

        # Store new memory
        record = {
            "user_id": user_id,
            "category": mem["category"],
            "content": mem["content"],
            "importance": mem.get("importance", 3),
            "active": True,
        }
        if source_turn_id:
            record["source_turn_id"] = source_turn_id

        # Try with confidence column; fall back without if column doesn't exist
        record_with_confidence = {**record, "confidence": mem.get("confidence", 0.5)}
        try:
            db.table("memories").insert(record_with_confidence).execute()
        except Exception as e:
            if "confidence" in str(e):
                logger.warning("memories.confidence column missing — storing without it")
                db.table("memories").insert(record).execute()
            else:
                raise

        stored += 1
        logger.info(f"Stored memory [{mem['category']}]: {mem['content'][:50]}")

    return stored


async def process_turn_memories(
    user_message: str,
    assistant_message: str,
    user_id: str,
    source_turn_id: str | None = None,
) -> list[dict[str, Any]]:
    """Full pipeline: extract → deduplicate → store. Fire-and-forget safe."""
    try:
        memories = await extract_memories(
            user_message, assistant_message, user_id, source_turn_id,
        )
        if memories:
            await deduplicate_and_store(memories, user_id, source_turn_id, user_message=user_message)
        return memories
    except Exception as e:
        logger.error(f"Memory processing failed: {e}")
        return []


def get_user_memories(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch active memories for a user, ordered by combined importance + recency score."""
    db = get_service_client()
    # Fetch more than limit to allow re-ranking
    fetch_limit = min(limit * 2, 200)
    result = (
        db.table("memories")
        .select("*")
        .eq("user_id", user_id)
        .eq("active", True)
        .order("importance", desc=True)
        .limit(fetch_limit)
        .execute()
    )
    memories = result.data or []

    # Re-rank with combined score: importance * 0.7 + recency * 0.3
    now = datetime.now(timezone.utc)
    for m in memories:
        importance = m.get("importance", 3)
        # Normalize importance to 0-1 scale (importance is 1-5)
        importance_norm = importance / 5.0
        recency = _recency_score(m.get("created_at"), now)
        m["_score"] = importance_norm * 0.7 + recency * 0.3

    memories.sort(key=lambda m: m.get("_score", 0), reverse=True)
    return memories[:limit]
