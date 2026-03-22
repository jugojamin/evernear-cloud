"""Tests for Opus codec — encode/decode round-trip and utilities."""

from __future__ import annotations
import struct

import pytest

from server.opus_codec import (
    decode_opus_frame,
    decode_opus_stream,
    encode_pcm_to_opus,
    is_opus_frame,
    reset,
)


@pytest.fixture(autouse=True)
def _reset_codec():
    """Reset singleton state between tests."""
    reset()
    yield
    reset()


class TestOpusRoundTrip:
    def test_encode_decode_silence(self):
        """Encode silence → decode → should get silence back."""
        frame_size = 480  # 20ms at 24kHz
        silence = struct.pack(f"<{frame_size}h", *([0] * frame_size))
        opus = encode_pcm_to_opus(silence, frame_size)
        pcm = decode_opus_frame(opus, frame_size)
        assert len(pcm) == len(silence)

    def test_encode_decode_tone(self):
        """Encode a simple tone → decode → verify length matches."""
        import math
        frame_size = 480
        samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(frame_size)]
        pcm_in = struct.pack(f"<{frame_size}h", *samples)
        opus = encode_pcm_to_opus(pcm_in, frame_size)
        pcm_out = decode_opus_frame(opus, frame_size)
        assert len(pcm_out) == len(pcm_in)

    def test_opus_compressed_smaller(self):
        """Opus output should be significantly smaller than PCM input."""
        frame_size = 480
        silence = struct.pack(f"<{frame_size}h", *([0] * frame_size))
        opus = encode_pcm_to_opus(silence, frame_size)
        assert len(opus) < len(silence), f"Opus ({len(opus)}) should be smaller than PCM ({len(silence)})"

    def test_decode_stream_multiple_frames(self):
        """Decode a stream of multiple Opus frames."""
        frame_size = 480
        frames = []
        for _ in range(5):
            silence = struct.pack(f"<{frame_size}h", *([0] * frame_size))
            frames.append(encode_pcm_to_opus(silence, frame_size))
        pcm = decode_opus_stream(frames, frame_size)
        expected_len = frame_size * 2 * 5  # 5 frames × 480 samples × 2 bytes
        assert len(pcm) == expected_len


class TestCodecDetection:
    def test_opus_frame_detected(self):
        """Small Opus-sized data should pass heuristic."""
        frame_size = 480
        silence = struct.pack(f"<{frame_size}h", *([0] * frame_size))
        opus = encode_pcm_to_opus(silence, frame_size)
        assert is_opus_frame(opus) is True

    def test_pcm_frame_not_opus(self):
        """Large PCM data should fail heuristic."""
        pcm = b"\x00" * 3840  # 40ms PCM at 48kHz
        assert is_opus_frame(pcm) is False

    def test_empty_not_opus(self):
        assert is_opus_frame(b"") is False

    def test_single_byte_not_opus(self):
        assert is_opus_frame(b"\x00") is False


class TestBackwardCompat:
    def test_pcm_passthrough(self):
        """PCM audio should not need Opus decoding — just verify it's not mangled."""
        pcm = struct.pack("<480h", *([1000] * 480))
        # PCM goes directly to Deepgram — no decode needed
        assert len(pcm) == 960  # 480 samples × 2 bytes


class TestGracefulDegradation:
    def test_corrupted_frame_inserts_silence(self):
        """Corrupted Opus frame should produce silence, not crash."""
        frame_size = 480
        silence = struct.pack(f"<{frame_size}h", *([0] * frame_size))
        good_frame = encode_pcm_to_opus(silence, frame_size)
        corrupted = b"\xff\xff\xff"  # Not valid Opus
        pcm = decode_opus_stream([good_frame, corrupted, good_frame], frame_size)
        expected_len = frame_size * 2 * 3
        assert len(pcm) == expected_len
