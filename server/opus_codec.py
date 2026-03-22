"""Opus codec utilities — server-side encode/decode for audio pipeline.

Provides Opus decoding (inbound: iOS → server) and encoding (outbound: server → iOS).
Used when Deepgram can't accept Opus directly, or for recording/monitoring PCM.

Deepgram nova-3 accepts Opus natively (encoding="opus"), so the primary audio
path forwards Opus frames directly. This module provides a fallback decode path
and utilities for testing/recording.
"""

from __future__ import annotations
import logging
import struct

logger = logging.getLogger(__name__)

# Lazy imports — opuslib requires libopus system library
_decoder = None
_encoder = None


def _get_decoder(sample_rate: int = 24000, channels: int = 1):
    """Get or create a singleton Opus decoder."""
    global _decoder
    if _decoder is None:
        import opuslib
        _decoder = opuslib.Decoder(sample_rate, channels)
    return _decoder


def _get_encoder(sample_rate: int = 24000, channels: int = 1):
    """Get or create a singleton Opus encoder."""
    global _encoder
    if _encoder is None:
        import opuslib
        _encoder = opuslib.Encoder(sample_rate, channels, "voip")
    return _encoder


def decode_opus_frame(opus_data: bytes, frame_size: int = 480) -> bytes:
    """Decode a single Opus frame to PCM16 (signed 16-bit LE).

    Args:
        opus_data: Raw Opus-encoded frame bytes
        frame_size: Number of samples per channel (480 = 20ms at 24kHz)

    Returns:
        PCM16 bytes (signed 16-bit little-endian, mono)
    """
    decoder = _get_decoder()
    return decoder.decode(opus_data, frame_size)


def encode_pcm_to_opus(pcm_data: bytes, frame_size: int = 480) -> bytes:
    """Encode PCM16 audio to a single Opus frame.

    Args:
        pcm_data: PCM16 bytes (signed 16-bit LE, mono, 24kHz)
        frame_size: Number of samples per channel (480 = 20ms at 24kHz)

    Returns:
        Opus-encoded frame bytes
    """
    encoder = _get_encoder()
    return encoder.encode(pcm_data, frame_size)


def decode_opus_stream(opus_frames: list[bytes], frame_size: int = 480) -> bytes:
    """Decode a sequence of Opus frames to continuous PCM16 audio.

    Args:
        opus_frames: List of raw Opus frame bytes
        frame_size: Samples per frame (480 = 20ms at 24kHz)

    Returns:
        Concatenated PCM16 bytes
    """
    decoder = _get_decoder()
    pcm_chunks = []
    for frame in opus_frames:
        try:
            pcm = decoder.decode(frame, frame_size)
            pcm_chunks.append(pcm)
        except Exception as e:
            logger.warning(f"Opus decode error on frame ({len(frame)} bytes): {e}")
            # Insert silence for corrupted frame (graceful degradation)
            silence = b"\x00" * (frame_size * 2)  # 2 bytes per sample (int16)
            pcm_chunks.append(silence)
    return b"".join(pcm_chunks)


def is_opus_frame(data: bytes) -> bool:
    """Heuristic check if data looks like an Opus frame.

    Opus frames are typically 3-200 bytes for speech. PCM frames at 48kHz
    are much larger (e.g., 3840 bytes for 40ms). This is a size-based
    heuristic — not definitive, but useful for auto-detection.
    """
    if len(data) < 2 or len(data) > 1275:  # Opus max frame is 1275 bytes
        return False
    # Opus TOC byte: top 5 bits encode config, bit 2 is stereo flag, bits 0-1 frame count
    # Valid configs: 0-31 (5 bits)
    toc = data[0]
    config = (toc >> 3) & 0x1F
    return config <= 31  # Always true for valid byte, but length check is the real filter


def reset():
    """Reset decoder/encoder state (e.g., between sessions)."""
    global _decoder, _encoder
    _decoder = None
    _encoder = None
