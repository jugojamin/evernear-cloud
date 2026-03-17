# EverNear Cloud Backend

Voice conversation server for EverNear — an AI companion for older adults.

## Architecture

```
iOS App ──WebSocket──► Fly.io Server ──► Deepgram (STT)
                                    ──► Claude Haiku/Sonnet (LLM)
                                    ──► Cartesia Sonic (TTS)
                                    ──► Supabase (DB + Auth)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (see .env.example)
cp .env.example .env
# Edit .env with your API keys

# Run locally
uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload

# Run tests
pytest tests/ -v
```

## Project Structure

```
server/
├── main.py              # FastAPI app + WebSocket endpoint
├── pipeline.py          # Pipecat voice pipeline
├── context.py           # System prompt + memories + history builder
├── memory.py            # Async memory extraction
├── auth.py              # JWT authentication
├── transport.py         # WebSocket connection manager
├── notifications.py     # APNs push notifications (stub)
├── metrics.py           # Per-turn latency tracking
├── config.py            # Environment configuration
├── routers/
│   ├── llm_router.py    # Haiku/Sonnet routing
│   └── tts_router.py    # Premium/standard TTS routing
├── db/
│   ├── client.py        # Supabase client
│   ├── models.py        # Pydantic data models
│   └── migrations/      # SQL migrations
└── prompts/
    ├── system_prompt.txt
    ├── memory_extraction.txt
    └── onboarding/      # 14 section prompts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/auth/signup` | Create account |
| POST | `/auth/login` | Login |
| POST | `/auth/refresh` | Refresh token |
| GET | `/user/profile` | Get profile |
| PATCH | `/user/profile` | Update profile |
| GET | `/user/memories` | List memories |
| GET | `/conversations` | List conversations |
| POST | `/consent` | Record consent |
| DELETE | `/user/data` | Delete all user data |
| WS | `/ws/voice` | Voice conversation |

## Deployment

```bash
# Deploy to Fly.io
fly deploy

# Set secrets
fly secrets set ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... \
  SUPABASE_URL=... SUPABASE_ANON_KEY=... SUPABASE_SERVICE_KEY=...
```

## Key Design Decisions

- **Latency target:** Sub-1-second end-to-end
- **LLM:** Claude Haiku 4.5 default, Sonnet for emotional/complex moments
- **TTS:** Cartesia Sonic with pluggable provider interface
- **VAD:** 1.5-2.0s silence threshold (elderly users)
- **Audio:** NEVER stored — deleted after STT transcription
- **Auth:** Supabase JWT with Row Level Security on all tables
- **Memory:** Async extraction after each turn, 11 categories
