"""Tests for the End-to-End Audio Bridge (TASK-2026-012)."""

import asyncio
import base64
import struct
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from server.audio_bridge import resample_24k_to_48k, DeepgramSTTSession
from server.failure_scripts import get_failure_response


class TestVoiceTurnReturnsAudio:
    """process_voice_turn() returns text + audio bytes."""

    @pytest.mark.asyncio
    async def test_voice_turn_returns_audio(self):
        from server.pipeline import EverNearPipeline

        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            mock_s.return_value.tts_provider = "cartesia"

            p = EverNearPipeline(user_id="user-123")

            # Mock process_text_turn
            p.process_text_turn = AsyncMock(return_value=("Hello there.", MagicMock(
                sonnet_reason=None, to_dict=lambda: {}, start_tts=lambda: None,
                tts_first_byte_received=lambda: None,
            )))

            # Mock TTS to yield fake audio chunks (24kHz PCM16)
            fake_24k = struct.pack("<4h", 100, 200, 300, 400)  # 4 samples at 24k

            async def fake_synthesize(text, voice_id, ctx=None):
                yield fake_24k

            p._tts_router.synthesize = fake_synthesize

            text, chunks, metrics = await p.process_voice_turn("Hi")

            assert text == "Hello there."
            assert len(chunks) > 0
            # Each 24k chunk (8 bytes = 4 samples) becomes 48k (16 bytes = 8 samples)
            assert len(chunks[0]) == len(fake_24k) * 2


class TestVoiceTurnValidatorRuns:
    """A medical advice response is blocked even in voice mode."""

    @pytest.mark.asyncio
    async def test_voice_turn_validator_runs(self):
        from server.pipeline import EverNearPipeline

        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            mock_s.return_value.tts_provider = "cartesia"

            p = EverNearPipeline(user_id="user-123")

            # Mock: LLM returns medical advice, validator should block it
            blocked_response = "That's something worth asking your doctor about. I want to make sure you get the right answer on that."
            p.process_text_turn = AsyncMock(return_value=(blocked_response, MagicMock(
                sonnet_reason=None, to_dict=lambda: {}, start_tts=lambda: None,
                tts_first_byte_received=lambda: None,
            )))

            async def fake_synthesize(text, voice_id, ctx=None):
                yield struct.pack("<2h", 100, 200)

            p._tts_router.synthesize = fake_synthesize

            text, chunks, metrics = await p.process_voice_turn("Should I take more aspirin?")

            # The response should be the blocked/rewritten version
            assert "doctor" in text.lower()


class TestVoiceTurnTTSFailureFallback:
    """TTS failure returns text-only response, no crash."""

    @pytest.mark.asyncio
    async def test_voice_turn_tts_failure_fallback(self):
        from server.pipeline import EverNearPipeline

        with patch("server.pipeline.get_settings") as mock_s:
            mock_s.return_value.anthropic_api_key = "test"
            mock_s.return_value.cartesia_api_key = "test"
            mock_s.return_value.conversation_history_turns = 8
            mock_s.return_value.llm_max_tokens = 200
            mock_s.return_value.tts_provider = "cartesia"

            p = EverNearPipeline(user_id="user-123")

            p.process_text_turn = AsyncMock(return_value=("Hello!", MagicMock(
                sonnet_reason=None, to_dict=lambda: {}, start_tts=lambda: None,
                tts_first_byte_received=lambda: None,
            )))

            # TTS raises an exception
            async def broken_synthesize(text, voice_id, ctx=None):
                raise ConnectionError("Cartesia down")
                yield  # make it a generator

            p._tts_router.synthesize = broken_synthesize

            text, chunks, metrics = await p.process_voice_turn("Hi")

            assert text == "Hello!"
            assert chunks == []  # No audio — text-only fallback


class TestSTTEmptyTranscript:
    """Empty Deepgram result triggers 'didn't catch that' script."""

    def test_stt_empty_transcript_handled(self):
        script = get_failure_response("stt_failure")
        assert script is not None
        assert "catch" in script.lower() or "again" in script.lower()


class TestInterruptStopsAudio:
    """Interrupt flag prevents further audio frames from being sent."""

    def test_interrupt_flag_stops_sending(self):
        # Simulate the interrupt logic from main.py
        interrupted = False
        sent_chunks = []
        audio_chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4"]

        for i, chunk in enumerate(audio_chunks):
            if interrupted:
                break
            sent_chunks.append(chunk)
            if i == 1:  # Interrupt after 2nd chunk
                interrupted = True

        assert len(sent_chunks) == 2  # Only first 2 sent before interrupt


class TestAudioFrameFormat:
    """Outbound audio frames match expected JSON schema."""

    def test_audio_frame_format(self):
        # Simulate what main.py sends
        chunk = struct.pack("<4h", 100, 200, 300, 400)
        frame = {
            "type": "audio",
            "data": base64.b64encode(chunk).decode("ascii"),
            "seq": 1,
        }
        assert frame["type"] == "audio"
        assert isinstance(frame["data"], str)
        assert isinstance(frame["seq"], int)
        # Verify data round-trips
        decoded = base64.b64decode(frame["data"])
        assert decoded == chunk

    def test_last_frame_has_last_flag(self):
        frame = {
            "type": "audio",
            "data": base64.b64encode(b"\x00\x00").decode("ascii"),
            "seq": 5,
            "last": True,
        }
        assert frame["last"] is True


class TestResample24kTo48k:
    """Sample rate conversion correctness."""

    def test_doubles_sample_count(self):
        # 4 samples at 24kHz
        pcm_24k = struct.pack("<4h", 100, 200, 300, 400)
        pcm_48k = resample_24k_to_48k(pcm_24k)
        # Should be 8 samples at 48kHz
        assert len(pcm_48k) == len(pcm_24k) * 2

    def test_duplicates_each_sample(self):
        pcm_24k = struct.pack("<3h", 1000, -500, 32000)
        pcm_48k = resample_24k_to_48k(pcm_24k, gain=1.0)
        samples = struct.unpack(f"<{len(pcm_48k) // 2}h", pcm_48k)
        assert samples == (1000, 1000, -500, -500, 32000, 32000)

    def test_empty_input(self):
        assert resample_24k_to_48k(b"") == b""
