"""EverNear Cloud Backend — FastAPI app + WebSocket endpoint."""

from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import logging
import time
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

# Server-level degraded flags — set when any user enters degraded mode,
# cleared when a successful response arrives.
stt_degraded_global = False
llm_degraded_global = False
llm_consecutive_failures = 0
llm_degraded_since = None  # float (time.monotonic) when degraded mode started
tts_degraded_global = False
tts_consecutive_failures = 0
tts_degraded_since = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EverNear Cloud Backend starting up")
    # Start background schedulers
    from server.checkin_scheduler import run_scheduler as checkin_scheduler
    from server.retention import run_scheduler as retention_scheduler
    checkin_task = asyncio.create_task(checkin_scheduler())
    retention_task = asyncio.create_task(retention_scheduler())
    yield
    checkin_task.cancel()
    retention_task.cancel()
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
    from fastapi.responses import JSONResponse

    deps = {
        "llm": "degraded" if llm_degraded_global else "ok",
        "stt": "degraded" if stt_degraded_global else "ok",
        "tts": "degraded" if tts_degraded_global else "ok",
    }
    degraded_list = [k for k, v in deps.items() if v == "degraded"]
    all_degraded = len(degraded_list) == 3
    status = "degraded" if degraded_list else "ok"

    body = {
        "status": status,
        "service": "evernear-api",
        "active_connections": manager.active_connections,
        "dependencies": deps,
        "degraded": degraded_list,
    }
    return JSONResponse(content=body, status_code=503 if all_degraded else 200)


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

    # Recent incidents (last 5, newest first)
    recent_incidents = [
        {
            "incident_id": e.get("incident_id"),
            "ts": e.get("ts"),
            "error_type": e.get("error_type"),
            "short_message": e.get("short_message"),
            "fallback_triggered": e.get("fallback_triggered"),
        }
        for e in reversed(entries[-5:])
    ]

    # Dependency health indicators
    def _dep_status(error_type: str) -> dict:
        dep_incidents = [e for e in entries if e.get("error_type") == error_type]
        dep_recent = [e for e in dep_incidents if _age(e) < 600]
        last_dep = dep_incidents[-1] if dep_incidents else None
        return {
            "status": "degraded" if dep_recent else "healthy",
            "last_failure": last_dep["ts"] if last_dep else None,
        }

    dependencies = {
        "llm": _dep_status("llm"),
        "stt": {
            "status": "degraded" if stt_degraded_global else _dep_status("stt")["status"],
            "last_failure": _dep_status("stt")["last_failure"],
        },
        "tts": _dep_status("tts"),
    }

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
        "stt_degraded": stt_degraded_global,
        "llm_degraded": llm_degraded_global,
        "tts_degraded": tts_degraded_global,
        "recent_incidents": recent_incidents,
        "dependencies": dependencies,
    }



@app.get("/api/incidents")
async def incidents_api():
    """Incident history for governance automation. Read-only."""
    entries = read_incidents()
    recent = list(reversed(entries[-10:]))
    return {"incidents": recent, "total": len(entries)}


# ─── Unit Economics ───────────────────────────────────────

@app.get("/api/unit-economics")
async def unit_economics(user_id: str | None = None, days: int | None = None):
    """Compute per-turn / per-conversation cost estimates from stored messages."""
    from datetime import timezone as _tz, timedelta
    settings = get_settings()
    db = get_service_client()

    # Build query — fetch messages joined with conversation user_id
    q = db.table("messages").select(
        "id, conversation_id, role, content, metrics, audio_duration_ms, created_at, "
        "conversations!inner(user_id, started_at)"
    )
    if user_id:
        q = q.eq("conversations.user_id", user_id)
    if days:
        since = (datetime.now(_tz.utc) - timedelta(days=days)).isoformat()
        q = q.gte("created_at", since)
    q = q.order("created_at", desc=False)

    try:
        result = q.execute()
    except Exception as e:
        logger.error(f"Unit economics query failed: {e}")
        return {"error": "query_failed", "detail": str(e)}

    rows = result.data or []
    if not rows:
        return {
            "summary": {
                "total_conversations": 0, "total_turns": 0,
                "total_cost_usd": 0.0, "avg_cost_per_conversation_usd": 0.0,
                "avg_cost_per_turn_usd": 0.0, "avg_turns_per_conversation": 0.0,
            },
            "breakdown": {"stt_total_usd": 0.0, "llm_total_usd": 0.0, "tts_total_usd": 0.0},
            "per_conversation": [],
        }

    # Group by conversation
    convos: dict[str, list[dict]] = {}
    for row in rows:
        cid = row["conversation_id"]
        convos.setdefault(cid, []).append(row)

    stt_total = llm_total = tts_total = 0.0
    per_convo = []

    for cid, msgs in convos.items():
        c_stt = c_llm = c_tts = 0.0
        turn_count = 0

        for msg in msgs:
            role = msg["role"]
            content = msg.get("content") or ""
            metrics = msg.get("metrics") or {}

            if role == "user":
                # STT cost: use audio_duration_ms if available, else estimate from stt_ms
                audio_ms = msg.get("audio_duration_ms") or metrics.get("stt_ms") or 0
                if audio_ms > 0:
                    c_stt += (audio_ms / 60000.0) * settings.deepgram_per_minute

            elif role == "assistant":
                turn_count += 1
                # LLM cost: estimate tokens from content length (chars / 4)
                est_output_tokens = len(content) / 4.0
                # Estimate input ~2x output for conversation context
                est_input_tokens = est_output_tokens * 2.0
                model = metrics.get("model_used", "haiku-4.5")
                if "sonnet" in model.lower():
                    c_llm += (est_input_tokens / 1000) * settings.anthropic_sonnet_input_per_1k
                    c_llm += (est_output_tokens / 1000) * settings.anthropic_sonnet_output_per_1k
                else:
                    c_llm += (est_input_tokens / 1000) * settings.anthropic_haiku_input_per_1k
                    c_llm += (est_output_tokens / 1000) * settings.anthropic_haiku_output_per_1k

                # TTS cost: character count
                c_tts += len(content) * settings.cartesia_per_character

        stt_total += c_stt
        llm_total += c_llm
        tts_total += c_tts
        total_cost = c_stt + c_llm + c_tts

        started_at = None
        if msgs and msgs[0].get("conversations"):
            started_at = msgs[0]["conversations"].get("started_at")

        per_convo.append({
            "conversation_id": cid,
            "turns": turn_count,
            "cost_usd": round(total_cost, 6),
            "stt_usd": round(c_stt, 6),
            "llm_usd": round(c_llm, 6),
            "tts_usd": round(c_tts, 6),
            "created_at": started_at or (msgs[0].get("created_at") if msgs else None),
        })

    grand_total = stt_total + llm_total + tts_total
    total_turns = sum(c["turns"] for c in per_convo)
    n_convos = len(per_convo)

    return {
        "summary": {
            "total_conversations": n_convos,
            "total_turns": total_turns,
            "total_cost_usd": round(grand_total, 6),
            "avg_cost_per_conversation_usd": round(grand_total / n_convos, 6) if n_convos else 0.0,
            "avg_cost_per_turn_usd": round(grand_total / total_turns, 6) if total_turns else 0.0,
            "avg_turns_per_conversation": round(total_turns / n_convos, 2) if n_convos else 0.0,
        },
        "breakdown": {
            "stt_total_usd": round(stt_total, 6),
            "llm_total_usd": round(llm_total, 6),
            "tts_total_usd": round(tts_total, 6),
        },
        "per_conversation": per_convo,
    }


# ─── Usage Analytics ─────────────────────────────────────

@app.get("/api/analytics")
async def analytics(days: int | None = None):
    """Usage analytics from conversations, messages, and audit_log. Read-only."""
    from datetime import timezone as _tz, timedelta
    db = get_service_client()
    now = datetime.now(_tz.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()
    since = (now - timedelta(days=days)).isoformat() if days else None

    try:
        # Conversations
        cq = db.table("conversations").select("id, user_id, started_at, turn_count")
        if since:
            cq = cq.gte("started_at", since)
        convos = (cq.execute()).data or []

        # Messages
        mq = db.table("messages").select("id, role, created_at, conversation_id")
        if since:
            mq = mq.gte("created_at", since)
        msgs = (mq.execute()).data or []

        # Audit log (safety interventions)
        aq = db.table("audit_log").select("id, created_at").eq("action", "response_validation")
        if since:
            aq = aq.gte("created_at", since)
        audits = (aq.execute()).data or []

    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        return {"error": "query_failed", "detail": str(e)}

    # Distinct users
    all_user_ids = set(c["user_id"] for c in convos if c.get("user_id"))
    today_users = set(c["user_id"] for c in convos if c.get("started_at", "") >= today_start)
    week_users = set(c["user_id"] for c in convos if c.get("started_at", "") >= week_ago)
    month_users = set(c["user_id"] for c in convos if c.get("started_at", "") >= month_ago)

    # Conversation stats
    today_convos = [c for c in convos if c.get("started_at", "") >= today_start]
    week_convos = [c for c in convos if c.get("started_at", "") >= week_ago]
    month_convos = [c for c in convos if c.get("started_at", "") >= month_ago]
    turn_counts = [c.get("turn_count", 0) or 0 for c in convos]
    avg_turns = round(sum(turn_counts) / len(turn_counts), 1) if turn_counts else 0.0

    # Avg duration: from started_at to last message created_at per conversation
    convo_ids = {c["id"] for c in convos}
    msg_by_convo: dict[str, list[str]] = {}
    for m in msgs:
        cid = m.get("conversation_id")
        if cid in convo_ids:
            msg_by_convo.setdefault(cid, []).append(m.get("created_at", ""))

    durations = []
    for c in convos:
        started = c.get("started_at", "")
        last_msgs = msg_by_convo.get(c["id"], [])
        if started and last_msgs:
            last_msg = max(last_msgs)
            try:
                from datetime import datetime as _dt
                s = _dt.fromisoformat(started)
                e = _dt.fromisoformat(last_msg)
                dur_min = (e - s).total_seconds() / 60.0
                if dur_min >= 0:
                    durations.append(dur_min)
            except Exception:
                pass
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0.0

    # Messages
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
    today_msgs = [m for m in msgs if m.get("created_at", "") >= today_start]

    # Safety
    week_audits = [a for a in audits if a.get("created_at", "") >= week_ago]

    return {
        "period": f"last_{days}d" if days else "all",
        "users": {
            "total": len(all_user_ids),
            "active_today": len(today_users),
            "active_7d": len(week_users),
            "active_30d": len(month_users),
        },
        "conversations": {
            "total": len(convos),
            "today": len(today_convos),
            "last_7d": len(week_convos),
            "last_30d": len(month_convos),
            "avg_turns": avg_turns,
            "avg_duration_minutes": avg_duration,
        },
        "messages": {
            "total": len(msgs),
            "user_messages": len(user_msgs),
            "assistant_messages": len(asst_msgs),
            "today": len(today_msgs),
        },
        "safety": {
            "total_interventions": len(audits),
            "last_7d": len(week_audits),
        },
    }


# ─── Check-In API ────────────────────────────────────────

@app.get("/api/checkins")
async def checkins_api():
    """Check-in event history + response rate. Read-only."""
    from server.checkin_log import get_checkins
    recent, rate = get_checkins(20)
    return {"checkins": recent, "total": len(recent), "response_rate": rate}


@app.post("/api/checkins/trigger")
async def trigger_checkin(body: dict | None = None):
    """Manually trigger a check-in for testing."""
    from server.checkin_scheduler import _fire_checkin
    schedule_name = (body or {}).get("schedule_name", "manual")
    await _fire_checkin(schedule_name)
    return {"status": "fired", "schedule_name": schedule_name}


# ─── Diagnostics ─────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_DIAGNOSTICS_FILE = _Path("/app/diagnostics.json")
_MAX_DIAGNOSTICS = 100


def _read_diagnostics() -> list[dict]:
    try:
        if _DIAGNOSTICS_FILE.exists():
            return _json.loads(_DIAGNOSTICS_FILE.read_text())
    except Exception:
        pass
    return []


def _write_diagnostics(entries: list[dict]):
    try:
        _DIAGNOSTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DIAGNOSTICS_FILE.write_text(_json.dumps(entries[-_MAX_DIAGNOSTICS:], indent=2))
    except Exception as e:
        logger.error(f"Failed to write diagnostics: {e}")


class DiagnosticReport(BaseModel):
    type: str  # crash, hang, cpu, metric
    timestamp: str
    summary: str
    raw: dict | None = None


@app.post("/api/diagnostics")
async def post_diagnostic(report: DiagnosticReport, user_id: str = Depends(require_auth)):
    """Receive crash/diagnostic reports from iOS MetricKit."""
    entry = {
        "user_id": user_id,
        "type": report.type,
        "timestamp": report.timestamp,
        "summary": report.summary,
        "raw": report.raw,
        "received_at": datetime.now(datetime.now().astimezone().tzinfo).isoformat(),
    }
    entries = _read_diagnostics()
    entries.append(entry)
    _write_diagnostics(entries)
    logger.info(f"Diagnostic received: type={report.type}, user={user_id}, summary={report.summary[:100]}")
    return {"status": "received"}


@app.get("/api/diagnostics")
async def get_diagnostics():
    """Recent diagnostic reports. Read-only."""
    entries = _read_diagnostics()
    recent = list(reversed(entries[-20:]))
    return {"diagnostics": recent, "total": len(entries)}


# ─── Retention ────────────────────────────────────────────

@app.get("/api/retention")
async def retention_status():
    """Current retention config + last cleanup run. Read-only."""
    from server.retention import get_retention_status
    return get_retention_status()


# ─── Quality Metrics ─────────────────────────────────────

@app.get("/api/quality-metrics")
async def quality_metrics(days: int | None = None):
    """Conversation quality metrics derived from existing message data. Read-only."""
    from datetime import timezone as _tz, timedelta
    db = get_service_client()
    since = (datetime.now(_tz.utc) - timedelta(days=days)).isoformat() if days else None

    try:
        # Conversations
        cq = db.table("conversations").select("id, user_id, started_at, turn_count")
        if since:
            cq = cq.gte("started_at", since)
        convos = (cq.execute()).data or []

        # Messages with timestamps
        mq = db.table("messages").select("id, conversation_id, role, content, created_at, latency_ms, metrics")
        if since:
            mq = mq.gte("created_at", since)
        mq = mq.order("created_at", desc=False)
        msgs = (mq.execute()).data or []
    except Exception as e:
        logger.error(f"Quality metrics query failed: {e}")
        return {"error": "query_failed", "detail": str(e)}

    if not msgs:
        return {
            "period": f"last_{days}d" if days else "all",
            "response_latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "sample_count": 0},
            "turn_length": {"avg_user_chars": 0, "avg_assistant_chars": 0, "ratio": 0},
            "session_health": {"avg_turns_per_session": 0, "single_turn_sessions": 0, "sessions_over_5_turns": 0, "avg_session_duration_seconds": 0},
            "drop_off": {"sessions_with_no_user_response": 0, "pct_sessions_under_2_turns": 0},
        }

    # Group messages by conversation
    msg_by_convo: dict[str, list[dict]] = {}
    for m in msgs:
        msg_by_convo.setdefault(m["conversation_id"], []).append(m)

    # Response latency: time between user msg and next assistant msg
    latencies = []
    for cid, cmsg in msg_by_convo.items():
        for i, m in enumerate(cmsg):
            if m["role"] == "user" and i + 1 < len(cmsg) and cmsg[i + 1]["role"] == "assistant":
                # Use latency_ms if available, else derive from timestamps
                lat = cmsg[i + 1].get("latency_ms")
                if lat and lat > 0:
                    latencies.append(lat)
                else:
                    try:
                        from datetime import datetime as _dt
                        t1 = _dt.fromisoformat(m["created_at"])
                        t2 = _dt.fromisoformat(cmsg[i + 1]["created_at"])
                        latencies.append(int((t2 - t1).total_seconds() * 1000))
                    except Exception:
                        pass

    latencies.sort()
    n_lat = len(latencies)
    avg_lat = int(sum(latencies) / n_lat) if n_lat else 0
    p50 = latencies[n_lat // 2] if n_lat else 0
    p95 = latencies[int(n_lat * 0.95)] if n_lat else 0

    # Turn length
    user_chars = [len(m.get("content", "")) for m in msgs if m["role"] == "user"]
    asst_chars = [len(m.get("content", "")) for m in msgs if m["role"] == "assistant"]
    avg_uc = round(sum(user_chars) / len(user_chars), 1) if user_chars else 0
    avg_ac = round(sum(asst_chars) / len(asst_chars), 1) if asst_chars else 0
    ratio = round(avg_ac / avg_uc, 1) if avg_uc > 0 else 0

    # Session health
    turn_counts = [c.get("turn_count", 0) or 0 for c in convos]
    avg_turns = round(sum(turn_counts) / len(turn_counts), 1) if turn_counts else 0
    single_turn = sum(1 for t in turn_counts if t <= 1)
    over_5 = sum(1 for t in turn_counts if t > 5)

    # Avg session duration
    durations_s = []
    for cid, cmsg in msg_by_convo.items():
        if len(cmsg) >= 2:
            try:
                from datetime import datetime as _dt
                first = _dt.fromisoformat(cmsg[0]["created_at"])
                last = _dt.fromisoformat(cmsg[-1]["created_at"])
                durations_s.append((last - first).total_seconds())
            except Exception:
                pass
    avg_dur_s = round(sum(durations_s) / len(durations_s), 1) if durations_s else 0

    # Drop-off
    n_convos = len(convos) or 1
    under_2 = sum(1 for t in turn_counts if t < 2)
    no_response = sum(1 for cid, cmsg in msg_by_convo.items() if all(m["role"] == "assistant" for m in cmsg))

    return {
        "period": f"last_{days}d" if days else "all",
        "response_latency": {"avg_ms": avg_lat, "p50_ms": p50, "p95_ms": p95, "sample_count": n_lat},
        "turn_length": {"avg_user_chars": avg_uc, "avg_assistant_chars": avg_ac, "ratio": ratio},
        "session_health": {
            "avg_turns_per_session": avg_turns,
            "single_turn_sessions": single_turn,
            "sessions_over_5_turns": over_5,
            "avg_session_duration_seconds": avg_dur_s,
        },
        "drop_off": {
            "sessions_with_no_user_response": no_response,
            "pct_sessions_under_2_turns": round(under_2 / n_convos, 2),
        },
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

from fastapi import Request

@app.delete("/user/data")
async def delete_user_data(request: Request, user_id: str = Depends(require_auth)):
    """Right to deletion (CCPA/MHMDA). FK-safe explicit table deletion with audit trail."""
    # Confirmation header required to prevent accidental deletion
    if request.headers.get("X-Confirm-Delete") != "true":
        raise HTTPException(400, {"error": "Deletion requires X-Confirm-Delete: true header"})

    db = get_service_client()
    now_iso = datetime.now(datetime.now().astimezone().tzinfo).isoformat()
    tables_to_clear = ["messages", "memories", "conversations", "consent_logs", "user_caregivers", "users"]
    completed = []

    # Write audit log BEFORE deletion (survives the delete)
    try:
        db.table("audit_log").insert({
            "user_id": user_id,
            "action": "data_deletion",
            "details": {"tables_cleared": tables_to_clear, "initiated_by": "user", "ts": now_iso},
        }).execute()
    except Exception as e:
        logger.error(f"Failed to write deletion audit log for {user_id}: {e}")
        # Continue anyway — deletion is more important than audit

    # FK-safe deletion order
    deletion_steps = [
        # 1. Messages (FK → conversations)
        ("messages", lambda: db.table("messages").delete().in_(
            "conversation_id",
            [c["id"] for c in (db.table("conversations").select("id").eq("user_id", user_id).execute()).data or []]
        ).execute()),
        # 2. Memories
        ("memories", lambda: db.table("memories").delete().eq("user_id", user_id).execute()),
        # 3. Conversations
        ("conversations", lambda: db.table("conversations").delete().eq("user_id", user_id).execute()),
        # 4. Consent logs
        ("consent_logs", lambda: db.table("consent_logs").delete().eq("user_id", user_id).execute()),
        # 5. User-caregiver links
        ("user_caregivers", lambda: db.table("user_caregivers").delete().eq("user_id", user_id).execute()),
        # 6. User record
        ("users", lambda: db.table("users").delete().eq("id", user_id).execute()),
    ]

    for table_name, delete_fn in deletion_steps:
        try:
            delete_fn()
            completed.append(table_name)
        except Exception as e:
            logger.error(f"Data deletion failed at {table_name} for {user_id}: {e}")
            log_incident("db", "main.py", f"Data deletion failed at {table_name}: {e}", fallback_triggered=False)
            return {
                "status": "partial_failure",
                "completed": completed,
                "failed_at": table_name,
                "error": str(e),
                "user_id": user_id,
            }

    logger.info(f"Full data deletion completed for user {user_id}")
    return {
        "status": "deleted",
        "user_id": user_id,
        "tables_cleared": completed,
        "tables_preserved": ["audit_log", "caregivers"],
        "deleted_at": now_iso,
    }


# ─── WebSocket Voice Endpoint ────────────────────────────

# ─── Rate Limiting ────────────────────────────────────────

import time as _time

_message_timestamps: dict[str, list[float]] = {}  # user_id -> list of timestamps


def _check_message_rate(user_id: str, limit: int) -> bool:
    """Return True if user is within rate limit, False if exceeded."""
    now = _time.time()
    window = 60.0
    timestamps = _message_timestamps.get(user_id, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= limit:
        _message_timestamps[user_id] = timestamps
        return False
    timestamps.append(now)
    _message_timestamps[user_id] = timestamps
    return True


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Main voice conversation WebSocket endpoint."""
    settings = get_settings()
    user_id = await authenticate_websocket(websocket)
    if not user_id:
        await websocket.accept()
        await websocket.close(code=4001, reason="Authentication required")
        return

    # Global connection limit
    if manager.active_connections >= settings.max_global_connections:
        await websocket.accept()
        logger.warning(f"Global rate limit: {manager.active_connections} active connections, rejecting new")
        await websocket.close(code=4503, reason="Service busy")
        return

    # Per-user connection limit
    if manager.user_connection_count(user_id) >= settings.max_connections_per_user:
        await websocket.accept()
        logger.warning(f"Rate limit: user {user_id} rejected — {manager.user_connection_count(user_id)} active connections")
        await websocket.close(code=4429, reason="Too many connections")
        return

    session = await manager.connect(websocket, user_id)
    pipeline = EverNearPipeline(user_id=user_id)
    logger.info(f"Session state for {user_id}: llm_degraded={llm_degraded_global}, stt_degraded={stt_degraded_global}, tts_degraded={tts_degraded_global}")

    # Audio state
    stt_session: DeepgramSTTSession | None = None
    interrupted = False
    sending_audio = False
    session_failure_count = 0
    transcript_timeout_task = None  # asyncio.Task | None
    stt_consecutive_failures = 0
    stt_degraded = False
    stt_degraded_since = None  # float (time.monotonic) when degraded mode started
    audio_frame_count = 0
    audio_codec = "pcm"  # "pcm" or "opus" — set by client in session_start

    async def _send_nudge_tts(text: str):
        """Send a short text as TTS audio to the user."""
        try:
            from server.audio_bridge import resample_24k_to_48k
            from server.routers.tts_router import TTSRoutingContext
            chunks = []
            async for chunk in pipeline._tts_router.synthesize(
                text=text, voice_id="", ctx=TTSRoutingContext()
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
            logger.error(f"TTS nudge failed: {tts_err}")

    async def _handle_voice_response(transcript: str):
        """Process a final STT transcript through the voice pipeline."""
        nonlocal interrupted, sending_audio, session_failure_count, stt_session
        nonlocal transcript_timeout_task, stt_consecutive_failures, stt_degraded, stt_degraded_since
        global stt_degraded_global

        # Mark any pending check-in as responded
        from server.checkin_log import mark_responded
        mark_responded(user_id)
        
        # Cancel transcript timeout — we got a response
        if transcript_timeout_task:
            transcript_timeout_task.cancel()
            transcript_timeout_task = None
        
        # Successful transcript — clear degraded mode and reset counter
        stt_consecutive_failures = 0
        if stt_degraded:
            logger.info(f"STT recovered for {user_id} — exiting degraded mode")
            stt_degraded = False
            stt_degraded_since = None
            stt_degraded_global = False
        
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

            # Per-user message rate limiting (skip for audio frames — only count actions)
            if msg_type in ("session_start", "text", "end_of_speech", "button_press"):
                if not _check_message_rate(user_id, settings.max_messages_per_minute):
                    logger.warning(f"Rate limit: user {user_id} exceeded {settings.max_messages_per_minute} messages/min")
                    await manager.send_json(user_id, {
                        "type": "transcript",
                        "text": "I need a moment to catch up — try again in a few seconds.",
                        "final": True,
                    })
                    continue

            if msg_type == "session_start":
                # Detect codec from client (default: pcm for backward compat)
                audio_codec = data.get("codec", "pcm")
                logger.info(f"Session start for {user_id}, codec={audio_codec}")
                # Start Deepgram STT session
                stt_session = DeepgramSTTSession(
                    on_transcript=_handle_voice_response,
                    codec=audio_codec,
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

                # Mark any pending check-in as responded
                from server.checkin_log import mark_responded
                mark_responded(user_id)

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
                    # Check degraded mode — skip Deepgram, respond immediately
                    if stt_degraded:
                        # Check if 2 minutes have passed — allow recovery attempt
                        if stt_degraded_since and (time.monotonic() - stt_degraded_since) >= 120:
                            logger.info(f"STT degraded mode recovery attempt for {user_id}")
                            stt_degraded = False
                            stt_degraded_since = None
                            # Fall through to normal STT path for retry
                        else:
                            # Still in degraded mode — respond immediately, no Deepgram
                            if not stt_session:  # only send once per button press (first audio frame)
                                nudge = "I'm still having trouble with voice right now. Try typing if you can, or give it a few minutes."
                                await manager.send_json(user_id, {"type": "transcript", "text": nudge, "final": True})
                                await _send_nudge_tts(nudge)
                                await manager.send_status(user_id, "listening")
                            continue
                    
                    # Create a fresh STT session if none exists (first frame of a new turn)
                    if not stt_session:
                        logger.info(f"Creating new Deepgram session for {user_id} (new turn, codec={audio_codec})")
                        stt_session = DeepgramSTTSession(
                            on_transcript=_handle_voice_response,
                            codec=audio_codec,
                        )
                        stt_started = await stt_session.start()
                        if not stt_started:
                            logger.error(f"Failed to create STT session for {user_id}")
                            stt_session = None
                    
                    if stt_session:
                        try:
                            pcm_bytes = base64.b64decode(audio_data)
                            audio_frame_count += 1
                            if audio_frame_count == 1:
                                logger.info(f"Audio stream started for {user_id}, first frame: {len(pcm_bytes)} bytes")
                            await stt_session.send_audio(pcm_bytes)
                        except Exception as e:
                            logger.error(f"Audio decode/forward error for {user_id}: {e}")

            elif msg_type == "end_of_speech":
                # User stopped talking — request final transcript
                if stt_session and stt_session.is_alive:
                    logger.info(f"end_of_speech received for {user_id} — {audio_frame_count} frames forwarded, calling finalize()")
                    audio_frame_count = 0
                    await stt_session.finalize()
                    
                    # Start a 10s timeout — if no transcript callback fires, nudge the user
                    async def _transcript_timeout():
                        nonlocal stt_consecutive_failures, stt_degraded, stt_degraded_since
                        global stt_degraded_global
                        await asyncio.sleep(10)
                        # If stt_session is still the same (not closed by a transcript callback), it timed out
                        if stt_session is not None:
                            stt_consecutive_failures += 1
                            logger.warning(f"Transcript timeout for {user_id} — no Deepgram response after 10s (consecutive: {stt_consecutive_failures})")
                            log_incident("stt", "main.py", f"Deepgram transcript timeout after 10s (consecutive: {stt_consecutive_failures})", fallback_triggered=True)
                            
                            if stt_consecutive_failures >= 3 and not stt_degraded:
                                # Enter degraded mode
                                stt_degraded = True
                                stt_degraded_since = time.monotonic()
                                stt_degraded_global = True
                                logger.warning(f"STT degraded mode activated for {user_id} after {stt_consecutive_failures} consecutive failures")
                                log_incident("stt", "main.py", f"STT degraded mode activated after {stt_consecutive_failures} failures", fallback_triggered=True)
                                nudge = "I'm having trouble hearing right now \u2014 it's not you, it's a technical issue on my end. You can try again in a few minutes, or type to me if you'd like to keep talking."
                            else:
                                nudge = "I didn't quite catch that \u2014 could you try again?"
                            
                            await manager.send_json(user_id, {"type": "transcript", "text": nudge, "final": True})
                            await _send_nudge_tts(nudge)
                            await manager.send_status(user_id, "listening")
                    
                    # Cancel any previous timeout task and start a new one
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
        logger.info(f"Session cleanup for {user_id}: stt={'open' if stt_session else 'closed'}, pending_timeout={transcript_timeout_task is not None}, frames={audio_frame_count}")
        if transcript_timeout_task:
            transcript_timeout_task.cancel()
            transcript_timeout_task = None
        if stt_session:
            await stt_session.close()
            stt_session = None
        audio_frame_count = 0
        manager.disconnect(user_id)
