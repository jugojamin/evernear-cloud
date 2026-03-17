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
    tts_voice_id: str = "ac317dac-1b8f-434f-b198-a490e2a4914d"  # Cartesia "Anneke - Trusted Guide" (candidate 1)

    # APNs (deferred)
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_key_path: str = ""

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
