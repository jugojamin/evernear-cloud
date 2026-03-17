"""Pydantic models for EverNear data layer."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


# --- Enums ---

class MemoryCategory(str, Enum):
    FAMILY = "family"
    HEALTH = "health"
    PREFERENCES = "preferences"
    STORIES = "stories"
    EMOTIONS = "emotions"
    MEANING = "meaning"
    CULTURE = "culture"
    FAITH = "faith"
    INTERESTS = "interests"
    CAREGIVERS = "caregivers"
    ROUTINE = "routine"


class ConsentType(str, Enum):
    HEALTH_DATA = "health_data"
    CAREGIVER_SHARING = "caregiver_sharing"
    RECORDING = "recording"
    DATA_RETENTION = "data_retention"


class ConsentMethod(str, Enum):
    VOICE = "voice"
    TAP = "tap"
    ONBOARDING = "onboarding"


class AuditAction(str, Enum):
    LOGIN = "login"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"
    MEMORY_CREATED = "memory_created"
    CAREGIVER_ALERT_SENT = "caregiver_alert_sent"
    RESPONSE_VALIDATION = "response_validation"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# --- Models ---

class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    email: str = ""
    display_name: str = ""
    preferred_name: str = ""
    voice_preference: str = ""
    onboarding_completed: bool = False
    onboarding_state: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    category: MemoryCategory
    content: str
    source_turn_id: UUID | None = None
    importance: int = 3  # 1-5
    confidence: float = 0.5  # 0.0-1.0 (1.0 = explicitly stated)
    verified: bool = False
    last_used_at: datetime | None = None
    correction_count: int = 0
    active: bool = True
    created_at: datetime | None = None


class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turn_count: int = 0
    summary: str = ""


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: MessageRole
    content: str
    audio_duration_ms: int | None = None
    latency_ms: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    sequence: int = 0
    created_at: datetime | None = None


class ConsentLog(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    consent_type: ConsentType
    granted: bool
    disclosure_version: str = "1.0"
    disclosure_hash: str = ""
    method: ConsentMethod = ConsentMethod.TAP
    ip_address: str = ""
    device_info: str = ""
    created_at: datetime | None = None
    revoked_at: datetime | None = None


class AuditEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    action: AuditAction
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str = ""
    created_at: datetime | None = None


class Caregiver(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    email: str = ""
    display_name: str = ""
    phone: str = ""
    created_at: datetime | None = None


class UserCaregiver(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    caregiver_id: UUID
    relationship: str = ""
    permissions: dict[str, Any] = Field(default_factory=dict)
    authorized_at: datetime | None = None
    authorized_via: str = "onboarding"
    active: bool = True
