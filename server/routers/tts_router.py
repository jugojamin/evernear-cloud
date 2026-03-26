"""TTS Router — pluggable provider interface with premium/standard routing."""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract TTS provider interface."""

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio bytes for the given text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class CartesiaTTSProvider(TTSProvider):
    """Cartesia Sonic TTS provider."""

    def __init__(self, api_key: str, default_voice_id: str = "", speed: float = 0.85, emotion: str = "calm"):
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        self._speed = speed
        self._emotion = emotion

    @property
    def name(self) -> str:
        return "cartesia"

    async def synthesize(self, text: str, voice_id: str = "") -> AsyncIterator[bytes]:
        """Synthesize via Cartesia Sonic streaming API."""
        import httpx

        vid = voice_id or self.default_voice_id
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": self.api_key,
                    "Cartesia-Version": "2025-04-16",
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": "sonic-3",
                    "transcript": text,
                    "voice": {"mode": "id", "id": vid},
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": 24000,
                    },
                    "generation_config": {
                        "speed": self._speed,
                        "emotion": self._emotion,
                    },
                },
                timeout=30.0,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk


@dataclass
class TTSRoutingContext:
    """Context for deciding premium vs standard TTS."""
    onboarding_active: bool = False
    emotional_response: bool = False  # True when Sonnet was used for LLM
    is_storytelling: bool = False
    first_conversation_today: bool = False
    is_reminder: bool = False
    is_confirmation: bool = False


class TTSRouter:
    """Routes text to appropriate TTS provider based on context."""

    def __init__(
        self,
        premium_provider: TTSProvider,
        standard_provider: TTSProvider | None = None,
    ):
        self.premium = premium_provider
        # For MVP, standard falls back to premium
        self.standard = standard_provider or premium_provider

    def select_provider(self, ctx: TTSRoutingContext) -> TTSProvider:
        """Select provider based on context. Premium for emotional/onboarding, standard otherwise."""
        if ctx.onboarding_active or ctx.emotional_response or ctx.is_storytelling or ctx.first_conversation_today:
            return self.premium
        if ctx.is_reminder or ctx.is_confirmation:
            return self.standard
        return self.premium  # Default premium for MVP

    async def synthesize(
        self, text: str, voice_id: str, ctx: TTSRoutingContext | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize text with the appropriate provider."""
        provider = self.select_provider(ctx or TTSRoutingContext())
        logger.info(f"TTS routing to {provider.name}")
        async for chunk in provider.synthesize(text, voice_id):
            yield chunk
