"""EverNear Cloud Backend — FastAPI app + WebSocket endpoint."""

from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.audio_bridge import DeepgramSTTSession
from server.auth import require_auth, authenticate_websocket
from server.config import get_settings
from server.incident_log import log_incident, read_incidents
from server.db.client import get_service_client
from server.failure_scripts import get_failure_response
from server.metrics import TurnMetrics
from server.notifications import push_service
from server.pipeline import EverNearPipeline
from server.transport import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EverNear Cloud Backend starting up")
    yield
    logger.info("EverNear Cloud Backend shutting down")


app = FastAPI(
    title="EverNear Cloud API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "evernear-api",
        "active_connections": manager.active_connections,
    }


@app.get("/api/runtime-status")
async def runtime_status():
    """Runtime health status for Mission Control."""
    from datetime import timezone as _tz

    entries = read_incidents()
    now = datetime.now(_tz.utc)

    def _age(e):
        return (now - datetime.fromisoformat(e["ts"])).total_seconds()

    recent = [e for e in entries if _age(e) < 600]
    recent_hour = [e for e in entries if _age(e) < 3600]

    if recent:
        status = "incident"
    elif recent_hour:
        status = "degraded"
    else:
        status = "healthy"

    last = entries[-1] if entries else None
    return {
        "status": status,
        "last_incident_ts": last["ts"] if last else None,
        "last_error_type": last.get("error_type") if last else None,
        "last_incident_id": last.get("incident_id") if last else None,
        "last_fallback_triggered": last.get("fallback_triggered") if last else None,
        "last_short_message": last.get("short_message") if last else None,
        "recent_failures_10m": len(recent),
        "recent_failures_1h": len(recent_hour),
        "total_logged": len(entries),
    }



# ─── Auth Endpoints ──────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""
    preferred_name: str = ""
    setup_type: str = "self"  # "self" or "caregiver"


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/signup")
async def signup(req: SignupRequest):
    db = get_service_client()
    try:
        # Create auth user via Supabase
        auth_resp = db.auth.sign_up({
            "email": req.email,
            "password": req.password,
        })
        user_id = auth_resp.user.id if auth_resp.user else None
        if not user_id:
            raise HTTPException(400, "Signup failed")

        # Create user profile
        db.table("users").insert({
            "id": str(user_id),
            "email": req.email,
            "display_name": req.display_name,
            "preferred_name": req.preferred_name or req.display_name,
            "onboarding_completed": False,
            "onboarding_state": {"current_section": "welcome", "completed": False},
            "settings": {},
        }).execute()

        # If caregiver setup, create caregiver record
        if req.setup_type == "caregiver":
            cg_id = str(uuid4())
            db.table("caregivers").insert({
                "id": cg_id,
                "email": req.email,
                "display_name": req.display_name,
            }).execute()
            db.table("user_caregivers").insert({
                "user_id": str(user_id),
                "caregiver_id": cg_id,
                "relationship": "other",
                "authorized_via": "onboarding",
            }).execute()

        # Audit log
        db.table("audit_log").insert({
            "user_id": str(user_id),
            "action": "login",
            "details": json.dumps({"event": "signup", "setup_type": req.setup_type}),
        }).execute()

        token = auth_resp.session.access_token if auth_resp.session else ""
        refresh = auth_resp.session.refresh_token if auth_resp.session else ""
        return {"user_id": str(user_id), "access_token": token, "refresh_token": refresh}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(500, f"Signup failed: {e}")


@app.post("/auth/login")
async def login(req: LoginRequest):
    db = get_service_client()
    try:
        auth_resp = db.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })
        user_id = auth_resp.user.id if auth_resp.user else None
        token = auth_resp.session.access_token if auth_resp.session else ""

        if user_id:
            db.table("audit_log").insert({
                "user_id": str(user_id),
                "action": "login",
                "details": json.dumps({"event": "login"}),
            }).execute()

        return {
            "user_id": str(user_id) if user_id else None,
            "access_token": token,
            "refresh_token": auth_resp.session.refresh_token if auth_resp.session else "",
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(401, "Invalid credentials")


@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    db = get_service_client()
    try:
        auth_resp = db.auth.refresh_session(refresh_token)
        return {
            "access_token": auth_resp.session.access_token if auth_resp.session else "",
            "refresh_token": auth_resp.session.refresh_token if auth_resp.session else "",
        }
    except Exception as e:
        raise HTTPException(401, f"Refresh failed: {e}")


# ─── User Profile ────────────────────────────────────────

@app.get("/user/profile")
async def get_profile(user_id: str = Depends(require_auth)):
    db = get_service_client()
    result = db.table("users").select("*").eq("id", user_id).single().execute()
    if not result.data:
        raise HTTPException(404, "User not found")
    return result.data


class ProfileUpdate(BaseModel):
    display_name: str | None = None
    preferred_name: str | None = None
    voice_preference: str | None = None
    settings: dict[str, Any] | None = None


@app.patch("/user/profile")
async def update_profile(update: ProfileUpdate, user_id: str = Depends(require_auth)):
    db = get_service_client()
    data = {k: v for k, v in update.model_dump().items() if v is not None}
    # Supabase JSONB accepts dicts directly — no json.dumps needed
    if not data:
        raise HTTPException(400, "No fields to update")
    db.table("users").update(data).eq("id", user_id).execute()
    return {"status": "updated"}


# ─── Memories ────────────────────────────────────────────

@app.get("/user/memories")
async def get_memories(user_id: str = Depends(require_auth)):
    db = get_service_client()
    result = (
        db.table("memories")
        .select("*")
        .eq("user_id", user_id)
        .eq("active", True)
        .order("importance", desc=True)
        .execute()
    )
    return result.data or []


# ─── Conversations ───────────────────────────────────────

@app.get("/conversations")
async def get_conversations(user_id: str = Depends(require_auth)):
    db = get_service_client()
    result = (
        db.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("started_at", desc=True)
        .limit(20)
        .execute()
    )
    return result.data or []


# ─── Consent ─────────────────────────────────────────────

class ConsentRequest(BaseModel):
    consent_type: str
    granted: bool
    disclosure_version: str = "1.0"
    disclosure_text: str = ""
    method: str = "tap"
    device_info: str = ""


@app.post("/consent")
async def record_consent(req: ConsentRequest, user_id: str = Depends(require_auth)):
    db = get_service_client()

    disclosure_hash = hashlib.sha256(req.disclosure_text.encode()).hexdigest() if req.disclosure_text else ""

    record = {
        "user_id": user_id,
        "consent_type": req.consent_type,
        "granted": req.granted,
        "disclosure_version": req.disclosure_version,
        "disclosure_hash": disclosure_hash,
        "method": req.method,
        "device_info": req.device_info,
    }
    db.table("consent_logs").insert(record).execute()

    # Audit
    action = "consent_granted" if req.granted else "consent_revoked"
    db.table("audit_log").insert({
        "user_id": user_id,
        "action": action,
        "details": json.dumps({"consent_type": req.consent_type, "version": req.disclosure_version}),
    }).execute()

    return {"status": "recorded", "consent_type": req.consent_type, "granted": req.granted}


# ─── Data Deletion ───────────────────────────────────────

@app.delete("/user/data")
async def delete_user_data(user_id: str = Depends(require_auth)):
    """Right to deletion (GDPR/CCPA). Cascading delete removes all user data."""
    db = get_service_client()

    # Audit before deletion
    db.table("audit_log").insert({
        "user_id": user_id,
        "action": "data_deleted",
        "details": json.dumps({"event": "user_requested_deletion"}),
    }).execute()

    # Cascade delete (FK constraints handle related records)
    db.table("users").delete().eq("id", user_id).execute()

    return {"status": "deleted", "message": "All user data has been permanently removed."}


# ─── WebSocket Voice Endpoint ────────────────────────────

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Main voice conversation WebSocket endpoint."""
    user_id = await authenticate_websocket(websocket)
    if not user_id:
        await websocket.accept()
        await websocket.close(code=4001, reason="Authentication required")
        return

    session = await manager.connect(websocket, user_id)
    pipeline = EverNearPipeline(user_id=user_id)

    # Audio state
    stt_session: DeepgramSTTSession | None = None
    interrupted = False
    sending_audio = False
    session_failure_count = 0
    transcript_timeout_task: asyncio.Task | None = None

    async def _handle_voice_response(transcript: str):
        """Process a final STT transcript through the voice pipeline."""
        nonlocal interrupted, sending_audio, session_failure_count, stt_session
        
        # Cancel transcript timeout — we got a response
        nonlocal transcript_timeout_task
        if transcript_timeout_task:
            transcript_timeout_task.cancel()
            transcript_timeout_task = None
        
        # Close STT session in background — don't await to avoid RecursionError
        # from Deepgram's async cleanup chain. A fresh session is created on next turn.
        if stt_session:
            logger.info(f"Closing STT session after transcript for {user_id}")
            _closing = stt_session
            stt_session = None
            asyncio.create_task(_closing.close())
        
        # Reset interrupt flag — this is a new turn
        interrupted = False

        if not transcript.strip():
            # Empty transcript — send failure script
            script = get_failure_response("stt_failure", session_failure_count)
            session_failure_count += 1
            await _send_failure_as_voice(script)
            return

        try:
            await manager.send_status(user_id, "thinking")

            response_text, audio_chunks, metrics = await pipeline.process_voice_turn(transcript)

            # LLM succeeded if we got here — send transcript for display
            await manager.send_json(user_id, {
                "type": "transcript",
                "text": response_text,
                "final": True,
                "metrics": metrics.to_dict(),
            })

            logger.info(f"Voice response for {user_id}: text={len(response_text)}ch, audio_chunks={len(audio_chunks) if audio_chunks else 0}, interrupted={interrupted}")
            
            if audio_chunks and not interrupted:
                # Send audio frames
                await manager.send_status(user_id, "speaking")
                sending_audio = True
                sent_count = 0

                for i, chunk in enumerate(audio_chunks):
                    if interrupted:
                        logger.info(f"Audio send interrupted at frame {i}")
                        break
                    is_last = (i == len(audio_chunks) - 1)
                    await manager.send_json(user_id, {
                        "type": "audio",
                        "data": base64.b64encode(chunk).decode("ascii"),
                        "seq": i + 1,
                        **({"last": True} if is_last else {}),
                    })
                    sent_count += 1

                sending_audio = False
                logger.info(f"Sent {sent_count} audio frames to {user_id}")
            elif not audio_chunks:
                logger.warning(f"No audio chunks from TTS for {user_id}")
            elif interrupted:
                logger.info(f"Skipped audio send — interrupted for {user_id}")

            # Back to listening
            interrupted = False
            await manager.send_status(user_id, "listening")

        except Exception as e:
            logger.error(f"Voice pipeline error for {user_id}: {e}")
            log_incident("transport", "main.py", f"Voice pipeline error: {e}", fallback_triggered=True)
            # Only fire failure scripts for actual LLM/pipeline errors,
            # not post-LLM DB writes (which are caught inside _store_message)
            session_failure_count += 1
            script = get_failure_response("llm_error", session_failure_count)
            await _send_failure_as_voice(script)

    async def _send_failure_as_voice(script: str | None):
        """Send a failure script as transcript (+ TTS if possible)."""
        nonlocal interrupted, sending_audio

        if not script:
            await manager.send_status(user_id, "listening")
            return

        try:
            # Always send transcript so user sees something
            await manager.send_json(user_id, {
                "type": "transcript",
                "text": script,
                "final": True,
            })

            # Try TTS for the failure script
            try:
                _, audio_chunks, _ = await pipeline.process_voice_turn.__wrapped__(
                    pipeline, script
                ) if hasattr(pipeline.process_voice_turn, '__wrapped__') else (script, [], None)
            except Exception:
                audio_chunks = []

            # Synthesize failure script audio directly via TTS router
            if not audio_chunks:
                try:
                    from server.audio_bridge import resample_24k_to_48k
                    from server.routers.tts_router import TTSRoutingContext

                    chunks = []
                    async for chunk in pipeline._tts_router.synthesize(
                        text=script, voice_id="", ctx=TTSRoutingContext()
                    ):
                        chunks.append(resample_24k_to_48k(chunk))
                    audio_chunks = chunks
                except Exception:
                    pass  # TTS failed for failure script — transcript only

            if audio_chunks and not interrupted:
                await manager.send_status(user_id, "speaking")
                sending_audio = True
                for i, chunk in enumerate(audio_chunks):
                    if interrupted:
                        break
                    is_last = (i == len(audio_chunks) - 1)
                    await manager.send_json(user_id, {
                        "type": "audio",
                        "data": base64.b64encode(chunk).decode("ascii"),
                        "seq": i + 1,
                        **({"last": True} if is_last else {}),
                    })
                sending_audio = False

        except Exception as e:
            logger.error(f"Failed to send failure script: {e}")
            log_incident("transport", "main.py", f"Failed to send failure script: {e}", fallback_triggered=False)
            # LAST RESORT: try bare transcript send
            try:
                await manager.send_json(user_id, {"type": "transcript", "text": "I'm here — just give me a moment.", "final": True})
            except Exception:
                pass  # WebSocket is truly dead — nothing more we can do

        interrupted = False
        await manager.send_status(user_id, "listening")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "session_start":
                # Start Deepgram STT session
                stt_session = DeepgramSTTSession(
                    on_transcript=_handle_voice_response,
                )
                stt_started = await stt_session.start()
                if not stt_started:
                    logger.warning(f"STT unavailable for {user_id} — text mode only")
                await manager.send_status(user_id, "listening")

            elif msg_type == "text":
                # Text mode (for testing without audio)
                text = data.get("text", "")
                if not text:
                    continue

                await manager.send_status(user_id, "thinking")
                try:
                    response, metrics = await pipeline.process_text_turn(text)
                except Exception as e:
                    logger.error(f"Text pipeline error for {user_id}: {e}")
                    log_incident("transport", "main.py", f"Text pipeline error: {e}", fallback_triggered=True)
                    response = "I'm having a little trouble right now — give me just a moment."
                    metrics = type('M', (), {'to_dict': lambda self: {}})()

                await manager.send_json(user_id, {
                    "type": "transcript",
                    "text": response,
                    "final": True,
                    "metrics": metrics.to_dict(),
                })
                await manager.send_status(user_id, "listening")

            elif msg_type == "audio":
                # Decode base64 PCM and forward to Deepgram
                audio_data = data.get("data", "")
                if audio_data:
                    # Create a fresh STT session if none exists (first frame of a new turn)
                    if not stt_session:
                        logger.info(f"Creating new Deepgram session for {user_id} (new turn)")
                        stt_session = DeepgramSTTSession(
                            on_transcript=_handle_voice_response,
                        )
                        stt_started = await stt_session.start()
                        if not stt_started:
                            logger.error(f"Failed to create STT session for {user_id}")
                            stt_session = None
                    
                    if stt_session:
                        try:
                            pcm_bytes = base64.b64decode(audio_data)
                            logger.info(f"Received audio frame from {user_id}, pcm_bytes={len(pcm_bytes)}")
                            await stt_session.send_audio(pcm_bytes)
                        except Exception as e:
                            logger.error(f"Audio decode/forward error for {user_id}: {e}")

            elif msg_type == "end_of_speech":
                # User stopped talking — request final transcript
                if stt_session and stt_session.is_alive:
                    logger.info(f"end_of_speech received for {user_id} — calling finalize()")
                    await stt_session.finalize()
                    
                    # Start a 10s timeout — if no transcript callback fires, nudge the user
                    async def _transcript_timeout():
                        await asyncio.sleep(10)
                        # If stt_session is still the same (not closed by a transcript callback), it timed out
                        if stt_session is not None:
                            logger.warning(f"Transcript timeout for {user_id} — no Deepgram response after 10s")
                            log_incident("stt", "main.py", "Deepgram transcript timeout after 10s", fallback_triggered=True)
                            nudge = "I didn't quite catch that — could you try again?"
                            await manager.send_json(user_id, {"type": "transcript", "text": nudge, "final": True})
                            # Send TTS nudge
                            try:
                                from server.audio_bridge import resample_24k_to_48k
                                from server.routers.tts_router import TTSRoutingContext
                                chunks = []
                                async for chunk in pipeline._tts_router.synthesize(
                                    text=nudge, voice_id="", ctx=TTSRoutingContext()
                                ):
                                    chunks.append(resample_24k_to_48k(chunk))
                                if chunks:
                                    await manager.send_status(user_id, "speaking")
                                    for i, chunk in enumerate(chunks):
                                        is_last = (i == len(chunks) - 1)
                                        await manager.send_json(user_id, {
                                            "type": "audio",
                                            "data": base64.b64encode(chunk).decode("ascii"),
                                            "seq": i + 1,
                                            **({"last": True} if is_last else {}),
                                        })
                            except Exception as tts_err:
                                logger.error(f"TTS for transcript timeout nudge failed: {tts_err}")
                            await manager.send_status(user_id, "listening")
                    
                    # Cancel any previous timeout task and start a new one
                    nonlocal transcript_timeout_task
                    if transcript_timeout_task:
                        transcript_timeout_task.cancel()
                    transcript_timeout_task = asyncio.create_task(_transcript_timeout())
                else:
                    logger.warning(f"end_of_speech but STT session dead for {user_id}")

            elif msg_type == "interrupt":
                # Barge-in — cancel current TTS streaming
                interrupted = True
                sending_audio = False
                logger.info(f"User {user_id} interrupted")
                await manager.send_status(user_id, "listening")

            elif msg_type == "session_end":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}")
    finally:
        if stt_session:
            await stt_session.close()
        manager.disconnect(user_id)
