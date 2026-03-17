"""In-memory cache of user's active memories for validator lookups."""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any

from server.db.client import get_service_client

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache of user's active memories for validator lookups.
    
    Provides sub-10ms lookups for the ResponseValidator.
    Load on session start, refresh after each memory extraction.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memories: dict[str, list[dict[str, Any]]] = {}  # keyed by category
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        """Load all active memories from DB. Call on session start."""
        try:
            db = get_service_client()
            result = (
                db.table("memories")
                .select("*")
                .eq("user_id", self.user_id)
                .eq("active", True)
                .execute()
            )
            self.memories = {}
            for m in result.data or []:
                cat = m.get("category", "general")
                self.memories.setdefault(cat, []).append(m)
            self.loaded_at = datetime.now()
            logger.info(f"MemoryCache loaded {sum(len(v) for v in self.memories.values())} memories for user {self.user_id}")
        except Exception as e:
            logger.error(f"MemoryCache load failed: {e}")
            self.memories = {}

    def refresh(self) -> None:
        """Reload after memory extraction. Call after each turn's extraction completes."""
        self.load()

    def lookup(self, category: str, content_substring: str) -> dict[str, Any] | None:
        """Find a matching memory by category and content substring.
        
        Returns memory dict or None.
        """
        content_lower = content_substring.lower()
        for m in self.memories.get(category, []):
            if content_lower in m.get("content", "").lower():
                return m
        return None

    def find_by_category_and_entities(
        self, category: str, entities: set[str]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Find a memory by category with entity overlap check.
        
        Returns (memory_or_none, has_overlap).
        - (memory, True) = matching memory with entity overlap
        - (memory, False) = same category but no entity overlap
        - (None, False) = no memory in this category
        """
        cat_memories = self.memories.get(category, [])
        if not cat_memories:
            return None, False

        entities_lower = {e.lower() for e in entities if len(e) > 1}
        
        for m in cat_memories:
            content_words = set(m.get("content", "").lower().split())
            overlap = entities_lower & content_words
            if overlap:
                return m, True

        # Category exists but no entity overlap
        return cat_memories[0], False

    def has_any(self, category: str) -> bool:
        """Check if any memories exist for a category."""
        return bool(self.memories.get(category))

    def get_all(self, category: str) -> list[dict[str, Any]]:
        """Get all memories for a category."""
        return self.memories.get(category, [])
