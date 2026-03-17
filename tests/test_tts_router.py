"""Tests for TTS Router — provider abstraction and routing logic."""

import pytest
from unittest.mock import AsyncMock
from server.routers.tts_router import (
    TTSProvider, TTSRouter, TTSRoutingContext, CartesiaTTSProvider,
)


class MockPremiumProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "premium_mock"

    async def synthesize(self, text: str, voice_id: str = ""):
        yield b"premium_audio_chunk"


class MockStandardProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "standard_mock"

    async def synthesize(self, text: str, voice_id: str = ""):
        yield b"standard_audio_chunk"


class TestTTSProviderInterface:
    def test_provider_has_name(self):
        p = MockPremiumProvider()
        assert p.name == "premium_mock"

    @pytest.mark.asyncio
    async def test_provider_synthesize_yields_bytes(self):
        p = MockPremiumProvider()
        chunks = []
        async for chunk in p.synthesize("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert isinstance(chunks[0], bytes)

    def test_cartesia_provider_has_name(self):
        p = CartesiaTTSProvider(api_key="test")
        assert p.name == "cartesia"


class TestTTSRouting:
    def setup_method(self):
        self.premium = MockPremiumProvider()
        self.standard = MockStandardProvider()
        self.router = TTSRouter(self.premium, self.standard)

    def test_onboarding_uses_premium(self):
        ctx = TTSRoutingContext(onboarding_active=True)
        assert self.router.select_provider(ctx).name == "premium_mock"

    def test_emotional_uses_premium(self):
        ctx = TTSRoutingContext(emotional_response=True)
        assert self.router.select_provider(ctx).name == "premium_mock"

    def test_storytelling_uses_premium(self):
        ctx = TTSRoutingContext(is_storytelling=True)
        assert self.router.select_provider(ctx).name == "premium_mock"

    def test_first_conversation_uses_premium(self):
        ctx = TTSRoutingContext(first_conversation_today=True)
        assert self.router.select_provider(ctx).name == "premium_mock"

    def test_reminder_uses_standard(self):
        ctx = TTSRoutingContext(is_reminder=True)
        assert self.router.select_provider(ctx).name == "standard_mock"

    def test_confirmation_uses_standard(self):
        ctx = TTSRoutingContext(is_confirmation=True)
        assert self.router.select_provider(ctx).name == "standard_mock"

    def test_default_uses_premium(self):
        ctx = TTSRoutingContext()
        assert self.router.select_provider(ctx).name == "premium_mock"

    def test_fallback_when_no_standard(self):
        router = TTSRouter(self.premium)  # No standard provider
        ctx = TTSRoutingContext(is_reminder=True)
        # Should fall back to premium
        assert router.select_provider(ctx).name == "premium_mock"

    @pytest.mark.asyncio
    async def test_synthesize_streams_bytes(self):
        chunks = []
        async for chunk in self.router.synthesize("Hello", "voice1"):
            chunks.append(chunk)
        assert len(chunks) > 0
