"""EverNear Cloud configuration via environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""

    # TTS
    tts_provider: str = "cartesia"  # cartesia | elevenlabs
    tts_voice_id: str = "6d287143-8db3-434a-959c-df147192da27"  # Cartesia "Stacy - Mentor" (candidate 9)
    tts_speed: float = 0.85  # Cartesia speed (0.6-1.5, default 1.0) — slowed for elderly users
    tts_emotion: str = "calm"  # Cartesia emotion baseline

    # APNs (deferred)
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_key_path: str = ""

    # Audio
    audio_gain: float = 2.0  # PCM gain multiplier for TTS output volume

    # API pricing (per-unit costs, USD) — updated 2026-03-19
    deepgram_per_minute: float = 0.0043          # Nova-3 pay-as-you-go
    anthropic_haiku_input_per_1k: float = 0.001  # Haiku 4.5 input
    anthropic_haiku_output_per_1k: float = 0.005 # Haiku 4.5 output
    anthropic_sonnet_input_per_1k: float = 0.003 # Sonnet input
    anthropic_sonnet_output_per_1k: float = 0.015  # Sonnet output
    cartesia_per_character: float = 0.000060     # Sonic-2 per character

    # Data retention (days, 0 = never expire)
    retention_messages_days: int = 365
    retention_incidents_days: int = 90
    retention_checkins_days: int = 90
    retention_diagnostics_days: int = 90
    retention_conversations_days: int = 365
    retention_memories_days: int = 0  # core to companion — never expire

    # Check-in scheduler
    checkin_enabled: bool = False  # OFF by default — flip when prompts are ready
    checkin_schedules: str = '[{"name": "morning", "hour": 9, "minute": 0, "tz": "America/Chicago"}, {"name": "evening", "hour": 20, "minute": 0, "tz": "America/Chicago"}]'

    # Rate limiting
    max_connections_per_user: int = 2
    max_messages_per_minute: int = 10
    max_global_connections: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # Pipeline tuning
    vad_silence_ms: int = 1800  # 1.8s for elderly users
    llm_max_tokens: int = 200
    max_context_tokens: int = 4000
    conversation_history_turns: int = 8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
