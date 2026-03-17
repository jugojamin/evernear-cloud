"""Audio Bridge — Deepgram async streaming STT + sample rate conversion.

Manages per-session Deepgram live transcription connections using the async SDK.
"""

from __future__ import annotations
import asyncio
import logging
import struct
from typing import Any, Callable, Awaitable

from deepgram import (
    AsyncListenWebSocketClient,
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

from server.config import get_settings
from server.failure_scripts import get_failure_response

logger = logging.getLogger(__name__)


def resample_24k_to_48k(pcm_24k: bytes) -> bytes:
    """Upsample PCM16 from 24kHz to 48kHz by duplicating each sample.

    Simple and fast — introduces no artifacts for speech audio.
    Input: PCM signed 16-bit LE at 24000 Hz
    Output: PCM signed 16-bit LE at 48000 Hz (double length)
    """
    sample_count = len(pcm_24k) // 2
    samples = struct.unpack(f"<{sample_count}h", pcm_24k[:sample_count * 2])
    upsampled = []
    for s in samples:
        upsampled.append(s)
        upsampled.append(s)
    return struct.pack(f"<{len(upsampled)}h", *upsampled)


class DeepgramSTTSession:
    """Manages an async streaming Deepgram STT session for one user.

    Uses the async WebSocket client (same as pipecat) to avoid
    threading conflicts with FastAPI's event loop.

    Usage:
        stt = DeepgramSTTSession(on_transcript=callback)
        await stt.start()
        await stt.send_audio(pcm_bytes)
        # ... callback fires with final transcripts
        await stt.close()
    """

    def __init__(
        self,
        on_transcript: Callable[[str], Awaitable[None]],
        on_interim: Callable[[str], Awaitable[None]] | None = None,
    ):
        self._on_transcript = on_transcript
        self._on_interim = on_interim
        self._connection: AsyncListenWebSocketClient | None = None
        self._client = None
        self._started = False
        self._transcript_buffer: list[str] = []

    async def start(self) -> bool:
        """Open a Deepgram async live transcription stream. Returns True on success."""
        s = get_settings()
        if not s.deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY not set — STT disabled")
            return False

        try:
            self._client = DeepgramClient(
                s.deepgram_api_key,
                config=DeepgramClientOptions(
                    options={"keepalive": "true"},
                ),
            )

            self._connection = self._client.listen.asyncwebsocket.v("1")

            # Register event handlers (async callbacks — no thread bridging needed)
            self._connection.on(LiveTranscriptionEvents.Open, self._handle_open)
            self._connection.on(LiveTranscriptionEvents.Transcript, self._handle_transcript)
            self._connection.on(LiveTranscriptionEvents.Error, self._handle_error)
            self._connection.on(LiveTranscriptionEvents.Close, self._handle_close)

            options = LiveOptions(
                model="nova-3",
                encoding="linear16",
                sample_rate=48000,
                interim_results=True,
                endpointing=s.vad_silence_ms,
                smart_format=True,
                punctuate=True,
            )

            result = await self._connection.start(options)
            if not result:
                logger.error("Deepgram async STT start() returned False")
                return False

            self._started = True
            logger.info(f"Deepgram async STT session started — is_connected={self._connection.is_connected}")
            return True

        except Exception as e:
            logger.error(f"Failed to start Deepgram async STT: {e}")
            return False

    async def _handle_open(self, *args, **kwargs) -> None:
        """Handle Deepgram connection open event."""
        logger.info("Deepgram async STT connection open — ready for audio")

    async def _handle_transcript(self, *args, **kwargs) -> None:
        """Handle transcript events from Deepgram (async — runs on event loop)."""
        try:
            result = kwargs.get("result", args[1] if len(args) > 1 else None)
            if result is None:
                return

            transcript = ""
            is_final = False

            if hasattr(result, "channel"):
                alt = result.channel.alternatives[0] if result.channel.alternatives else None
                if alt:
                    transcript = alt.transcript
                    is_final = result.is_final if hasattr(result, "is_final") else False
            elif isinstance(result, dict):
                channel = result.get("channel", {})
                alts = channel.get("alternatives", [])
                if alts:
                    transcript = alts[0].get("transcript", "")
                is_final = result.get("is_final", False)

            if not transcript:
                return

            if is_final:
                self._transcript_buffer.append(transcript)
                full = " ".join(self._transcript_buffer)
                self._transcript_buffer = []
                if full.strip():
                    logger.info(f"Deepgram final transcript: '{full.strip()}'")
                    # Direct async call — no thread bridging needed
                    await self._on_transcript(full.strip())
            elif self._on_interim:
                await self._on_interim(transcript)

        except Exception as e:
            logger.error(f"Transcript handler error: {e}")

    async def _handle_error(self, *args, **kwargs) -> None:
        """Handle Deepgram errors."""
        error = kwargs.get("error", args[1] if len(args) > 1 else "unknown")
        logger.error(f"Deepgram async STT error: {error}")
        self._started = False

    async def _handle_close(self, *args, **kwargs) -> None:
        """Handle Deepgram connection close."""
        logger.info("Deepgram async STT connection closed")
        self._started = False

    @property
    def is_alive(self) -> bool:
        """Check if the Deepgram session is still connected."""
        if not self._connection or not self._started:
            return False
        try:
            return self._connection.is_connected
        except Exception:
            return self._started

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send PCM audio bytes to the Deepgram stream (async)."""
        if self._connection and self._started:
            try:
                await self._connection.send(pcm_bytes)
                logger.info(f"Forwarded {len(pcm_bytes)} bytes to Deepgram")
            except Exception as e:
                logger.error(f"Failed to send audio to Deepgram: {e}")
                self._started = False

    async def finalize(self) -> None:
        """Signal end of speech — request final transcript."""
        if self._connection and self._started:
            try:
                await self._connection.finalize()
            except Exception as e:
                logger.error(f"Failed to finalize Deepgram stream: {e}")

    async def close(self) -> None:
        """Close the Deepgram stream by dropping references.
        
        We intentionally do NOT call finish() — the Deepgram async SDK's
        internal task cleanup triggers RecursionError/CancelledError that
        corrupts the asyncio event loop and blocks subsequent LLM calls.
        Nulling refs lets Python GC clean up the WebSocket naturally.
        """
        logger.info("Deepgram STT session closed (refs dropped, GC cleanup)")
        self._started = False
        self._connection = None
        self._client = None
