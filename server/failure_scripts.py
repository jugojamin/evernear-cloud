"""Pre-written failure response scripts — all static strings, never LLM-generated."""

from __future__ import annotations

FAILURE_SCRIPTS: dict[str, str | None] = {
    "api_timeout": "I lost my train of thought for a moment. What were you saying?",
    "stt_failure": "I didn't quite catch that. Could you say it again?",
    "llm_error": "My mind went blank for a second. I'm here — go ahead.",
    "tts_failure": None,  # Fall back to text in transcript
    "websocket_disconnect": "I'm having a little trouble hearing you. Give me just a moment.",
    "repeated_failure": "I'm not working quite right at the moment. I'll be back to normal soon.",
    "complete_outage": "I'm having trouble connecting right now. I'll be right here when you're back online.",
    "validator_failure": "Give me just a moment — I lost my train of thought. What were you saying?",
    "validator_timeout": "Give me just a moment — I lost my train of thought. What were you saying?",
    "memory_cache_failure": "I want to make sure I'm remembering that right. Can you tell me again?",
}

# Repeated failure threshold
REPEATED_FAILURE_THRESHOLD = 3


def get_failure_response(failure_type: str, session_failure_count: int = 0) -> str | None:
    """Get the appropriate failure response for a given failure type.
    
    Returns None for tts_failure (fall back to text).
    Returns repeated_failure script if session has exceeded threshold.
    """
    if session_failure_count >= REPEATED_FAILURE_THRESHOLD:
        return FAILURE_SCRIPTS["repeated_failure"]
    return FAILURE_SCRIPTS.get(failure_type, FAILURE_SCRIPTS["validator_failure"])
