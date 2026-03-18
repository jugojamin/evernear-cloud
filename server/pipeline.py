"""Pipecat voice conversation pipeline for EverNear.

Orchestrates: audio → STT → context → LLM → TTS → audio out.
Memory extraction runs asynchronously after each turn.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

import anthropic

from server.config import get_settings
from server.context import build_context
from server.memory import process_turn_memories, get_user_memories
from server.memory_cache import MemoryCache
from server.metrics import TurnMetrics
from server.audio_bridge import resample_24k_to_48k
from server.failure_scripts import get_failure_response
from server.routers.llm_router import route_llm, RoutingDecision
from server.routers.tts_router import CartesiaTTSProvider, TTSRouter, TTSRoutingContext
from server.validator import ResponseValidator, SessionContext, ValidationAction
from server.incident_log import log_incident
from server.db.client import get_service_client

logger = logging.getLogger(__name__)


class EverNearPipeline:
    """Voice conversation pipeline using streaming STT → LLM → TTS.

    In production with Pipecat, this orchestrates frame processors.
    For text-mode testing, call process_text_turn() directly.
    """

    def __init__(self, user_id: str, conversation_id: str | None = None):
        self.user_id = user_id
        self.conversation_id = conversation_id or str(uuid4())
        self.turn_count = 0
        self.conversation_history: list[dict[str, str]] = []
        self.last_memory_categories: list[str] = []

        # Safety: response validator + memory cache
        self._validator = ResponseValidator()
        self._memory_cache = MemoryCache(user_id)
        self._session_context = SessionContext(user_id=user_id)
        self._validator_timeout_ms = 50

        s = get_settings()
        self._anthropic = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)

        # TTS setup
        tts_provider = CartesiaTTSProvider(api_key=s.cartesia_api_key, default_voice_id=s.tts_voice_id)
        self._tts_router = TTSRouter(premium_provider=tts_provider)
        self._settings = s

    async def _get_user_profile(self) -> dict[str, Any]:
        """Fetch user profile from DB."""
        db = get_service_client()
        try:
            result = db.table("users").select("*").eq("id", self.user_id).execute()
            rows = result.data or []
            if rows:
                return rows[0]
            logger.warning(f"No user profile found for {self.user_id}")
            return {"preferred_name": "friend"}
        except Exception as e:
            logger.error(f"Failed to fetch user profile: {e}")
            return {"preferred_name": "friend"}

    async def process_text_turn(self, user_text: str) -> tuple[str, TurnMetrics]:
        """Process a text-mode conversation turn (no audio).

        Returns (assistant_response, metrics).
        """
        metrics = TurnMetrics()
        metrics.start_turn()
        self.turn_count += 1

        # Ensure conversation record exists (prevents FK constraint violations on messages)
        await self._ensure_conversation()

        # Fetch user profile and memories
        profile = await self._get_user_profile()
        user_name = profile.get("preferred_name", "")

        # Load memory cache and caregiver context on first turn
        if self.turn_count == 1:
            try:
                self._memory_cache.load()
                # Load caregiver name for Tier A responses
                caregiver_name = await self._get_caregiver_name()
                self._session_context.caregiver_name = caregiver_name
            except Exception as e:
                logger.error(f"Failed to load session context: {e}")
        onboarding_state = profile.get("onboarding_state", {})
        # Defensive: handle stringified JSON from older signups
        if isinstance(onboarding_state, str):
            try:
                onboarding_state = json.loads(onboarding_state)
            except (json.JSONDecodeError, TypeError):
                onboarding_state = {}
        memories = get_user_memories(self.user_id)

        # STT already done (text mode) — mark it
        metrics.stt_ms = 0

        # Route LLM
        onboarding_active = bool(
            onboarding_state and not onboarding_state.get("completed", False)
        )
        routing = route_llm(
            transcript=user_text,
            onboarding_active=onboarding_active,
            turn_count=self.turn_count,
            last_memory_categories=self.last_memory_categories,
        )
        metrics.model_used = routing.model
        metrics.sonnet_reason = routing.reason

        # Build context
        system_prompt, history = build_context(
            user_name=user_name,
            memories=memories,
            conversation_history=self.conversation_history,
            onboarding_state=onboarding_state,
            max_turns=self._settings.conversation_history_turns,
        )

        # Add user message to history
        history.append({"role": "user", "content": user_text})

        # Call Claude (streaming) with 1 retry
        metrics.start_llm()
        full_response = ""
        _FALLBACK_RESPONSE = "I'm having a little trouble right now — give me just a moment."
        _MAX_RETRIES = 1

        logger.info(f"Calling Claude {routing.model} for user {self.user_id}, input: '{user_text[:50]}'")
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                full_response = ""
                async with asyncio.timeout(30):
                    async with self._anthropic.messages.stream(
                        model=routing.model,
                        max_tokens=self._settings.llm_max_tokens,
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=history,
                    ) as stream:
                        first_token = True
                        async for text in stream.text_stream:
                            if first_token:
                                metrics.llm_first_token_received()
                                first_token = False
                            full_response += text
                last_error = None
                break  # success
            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"Claude API timeout (attempt {attempt + 1}) for {self.user_id}")
            except Exception as e:
                last_error = e
                logger.warning(f"Claude API error (attempt {attempt + 1}) for {self.user_id}: {type(e).__name__}: {e}")

            if attempt < _MAX_RETRIES:
                logger.info(f"Retrying Claude API call in 2s for {self.user_id}")
                await asyncio.sleep(2)

        if last_error is not None:
            error_class = "timeout" if last_error == "timeout" else type(last_error).__name__
            logger.error(f"Claude API failed after {_MAX_RETRIES + 1} attempts for {self.user_id}: {error_class}")
            full_response = _FALLBACK_RESPONSE
            # Log incident for Mission Control
            log_incident("llm", "pipeline.py", f"Claude API {error_class} after {_MAX_RETRIES + 1} attempts", fallback_triggered=True)

        metrics.end_llm()
        logger.info(f"Claude response for {self.user_id}: {len(full_response)} chars in {metrics.llm_total_ms}ms")

        # Response Validation — rule-based safety gate (no LLM calls)
        try:
            validation = self._validator.validate(
                response=full_response,
                user_input=user_text,
                memory_cache=self._memory_cache,
                session_context=self._session_context,
            )

            if validation.action != ValidationAction.ALLOW:
                full_response = validation.response

            # Log intervention (fire and forget)
            asyncio.create_task(self._log_validation(validation))

        except Exception as e:
            # Fail closed — block the response
            logger.error(f"Validator failed: {e}")
            log_incident("validator", "pipeline.py", f"Validator failed: {e}", fallback_triggered=True)
            full_response = ResponseValidator.UNIVERSAL_FALLBACK

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        # Store message in DB
        await self._store_message(user_text, "user", metrics)
        msg_id = await self._store_message(full_response, "assistant", metrics)

        # Async memory extraction (fire and forget)
        asyncio.create_task(
            self._extract_memories(user_text, full_response, msg_id)
        )

        metrics.end_turn()
        metrics.check_alerts()

        return full_response, metrics

    async def process_voice_turn(
        self, user_text: str
    ) -> tuple[str, list[bytes], TurnMetrics]:
        """Process a voice conversation turn — text pipeline + TTS.

        Returns (response_text, audio_chunks_48k, metrics).
        Audio chunks are PCM16 at 48kHz for iOS playback.
        """
        # Run the text pipeline (includes LLM + validator)
        response_text, metrics = await self.process_text_turn(user_text)

        # Synthesize TTS
        audio_chunks: list[bytes] = []
        try:
            metrics.start_tts()
            voice_id = self._settings.tts_provider  # placeholder — voice selection is next task
            # Use a known Cartesia voice ID or empty for default
            tts_voice = ""  # Let provider use its default

            first_chunk = True
            async for chunk in self._tts_router.synthesize(
                text=response_text,
                voice_id=tts_voice,
                ctx=TTSRoutingContext(
                    emotional_response=bool(metrics.sonnet_reason),
                ),
            ):
                if first_chunk:
                    metrics.tts_first_byte_received()
                    first_chunk = False
                # Resample 24kHz → 48kHz for iOS playback
                chunk_48k = resample_24k_to_48k(chunk)
                audio_chunks.append(chunk_48k)

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            log_incident("tts", "pipeline.py", f"TTS failed: {e}", fallback_triggered=False)
            # TTS failure — return text only (Safety Architecture Rule #6)
            # audio_chunks stays empty, caller sends transcript only

        return response_text, audio_chunks, metrics

    async def _store_message(
        self, content: str, role: str, metrics: TurnMetrics
    ) -> str | None:
        """Store a message in the DB. Returns message ID."""
        try:
            db = get_service_client()
            record = {
                "conversation_id": self.conversation_id,
                "role": role,
                "content": content,
                "sequence": self.turn_count * 2 + (0 if role == "user" else 1),
                "metrics": metrics.to_dict() if role == "assistant" else {},
                "latency_ms": metrics.total_ms if role == "assistant" else None,
            }
            result = db.table("messages").insert(record).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            log_incident("db", "pipeline.py", f"Failed to store message: {e}", fallback_triggered=False)
            return None

    async def _extract_memories(
        self, user_text: str, assistant_text: str, source_turn_id: str | None
    ):
        """Fire-and-forget memory extraction."""
        try:
            memories = await process_turn_memories(
                user_text, assistant_text, self.user_id, source_turn_id,
            )
            self.last_memory_categories = [m["category"] for m in memories]
            # Refresh memory cache after new memories stored
            if memories:
                try:
                    self._memory_cache.refresh()
                except Exception as e:
                    logger.error(f"Memory cache refresh failed: {e}")
        except Exception as e:
            logger.error(f"Background memory extraction failed: {e}")
            log_incident("db", "pipeline.py", f"Memory extraction failed: {e}", fallback_triggered=False)

    async def _get_caregiver_name(self) -> str:
        """Fetch primary caregiver display name for this user."""
        try:
            db = get_service_client()
            result = (
                db.table("user_caregivers")
                .select("caregiver_id, caregivers(display_name)")
                .eq("user_id", self.user_id)
                .eq("active", True)
                .limit(1)
                .execute()
            )
            if result.data and result.data[0].get("caregivers"):
                return result.data[0]["caregivers"].get("display_name", "")
        except Exception as e:
            logger.error(f"Failed to fetch caregiver: {e}")
        return ""

    async def _log_validation(self, result) -> None:
        """Log validation result to audit_log. Fire and forget."""
        try:
            record = {
                "user_id": self.user_id,
                "action": "response_validation",
                "details": {
                    "validation_action": result.action.value,
                    "checks_triggered": result.checks_triggered,
                    "latency_ms": result.latency_ms,
                },
            }
            if result.action != ValidationAction.ALLOW:
                record["details"]["original_response"] = result.original_response
                record["details"]["delivered_response"] = result.response
                record["details"]["user_input_excerpt"] = result.user_input[:200]

            db = get_service_client()
            db.table("audit_log").insert(record).execute()
        except Exception as e:
            logger.error(f"Validation log failed: {e}")
            log_incident("db", "pipeline.py", f"Validation log failed: {e}", fallback_triggered=False)

    async def _ensure_conversation(self):
        """Ensure conversation record exists in DB."""
        try:
            db = get_service_client()
            db.table("conversations").upsert({
                "id": self.conversation_id,
                "user_id": self.user_id,
                "turn_count": self.turn_count,
            }).execute()
        except Exception as e:
            logger.error(f"Failed to upsert conversation: {e}")
            log_incident("db", "pipeline.py", f"Failed to upsert conversation: {e}", fallback_triggered=False)
