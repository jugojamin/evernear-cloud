"""Memory Extractor — async post-turn fact extraction and deduplication."""

from __future__ import annotations
import asyncio
import json
import logging
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


async def deduplicate_and_store(
    memories: list[dict[str, Any]],
    user_id: str,
    source_turn_id: str | None = None,
) -> int:
    """Deduplicate against existing memories and store new ones. Returns count stored."""
    if not memories:
        return 0

    db = get_service_client()
    stored = 0

    for mem in memories:
        # Simple dedup: check for exact content match in same category
        existing = (
            db.table("memories")
            .select("id")
            .eq("user_id", user_id)
            .eq("category", mem["category"])
            .eq("content", mem["content"])
            .eq("active", True)
            .execute()
        )

        if existing.data:
            logger.debug(f"Duplicate memory skipped: {mem['content'][:50]}")
            continue

        # Store new memory
        record = {
            "user_id": user_id,
            "category": mem["category"],
            "content": mem["content"],
            "importance": mem.get("importance", 3),
            "confidence": mem.get("confidence", 0.5),
            "active": True,
        }
        if source_turn_id:
            record["source_turn_id"] = source_turn_id

        db.table("memories").insert(record).execute()
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
            await deduplicate_and_store(memories, user_id, source_turn_id)
        return memories
    except Exception as e:
        logger.error(f"Memory processing failed: {e}")
        return []


def get_user_memories(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch active memories for a user, ordered by importance."""
    db = get_service_client()
    result = (
        db.table("memories")
        .select("*")
        .eq("user_id", user_id)
        .eq("active", True)
        .order("importance", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
