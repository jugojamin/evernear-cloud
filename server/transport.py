"""WebSocket connection manager for EverNear voice sessions."""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class VoiceSession:
    """Represents an active WebSocket voice session."""
    user_id: str
    websocket: WebSocket
    conversation_id: str | None = None
    turn_count: int = 0
    connected: bool = True


class ConnectionManager:
    """Manages per-user WebSocket connections with rate limiting."""

    def __init__(self):
        self._sessions: dict[str, VoiceSession] = {}
        self._user_connections: dict[str, int] = {}  # user_id -> active count

    def user_connection_count(self, user_id: str) -> int:
        return self._user_connections.get(user_id, 0)

    async def connect(self, websocket: WebSocket, user_id: str) -> VoiceSession:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        # Disconnect existing session for this user (single-session per user)
        if user_id in self._sessions:
            old = self._sessions[user_id]
            old.connected = False
            try:
                await old.websocket.close(code=1000, reason="New session started")
            except Exception:
                pass
            # Count stays — will be reused by new session
        session = VoiceSession(user_id=user_id, websocket=websocket)
        self._sessions[user_id] = session
        self._user_connections[user_id] = self._user_connections.get(user_id, 0) + 1
        logger.info(f"User {user_id} connected via WebSocket (connections: {self._user_connections[user_id]})")
        return session

    def disconnect(self, user_id: str):
        """Remove a disconnected session."""
        session = self._sessions.pop(user_id, None)
        if session:
            session.connected = False
            count = self._user_connections.get(user_id, 1) - 1
            if count <= 0:
                self._user_connections.pop(user_id, None)
            else:
                self._user_connections[user_id] = count
            logger.info(f"User {user_id} disconnected (connections: {max(count, 0)})")

    def get_session(self, user_id: str) -> VoiceSession | None:
        return self._sessions.get(user_id)

    def is_connected(self, user_id: str) -> bool:
        session = self._sessions.get(user_id)
        return session is not None and session.connected

    @property
    def active_connections(self) -> int:
        return len(self._sessions)

    async def send_json(self, user_id: str, data: dict[str, Any]):
        """Send JSON message to a connected user."""
        session = self._sessions.get(user_id)
        if session and session.connected:
            try:
                await session.websocket.send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")
                self.disconnect(user_id)

    async def send_bytes(self, user_id: str, data: bytes):
        """Send binary audio data to a connected user."""
        session = self._sessions.get(user_id)
        if session and session.connected:
            try:
                await session.websocket.send_bytes(data)
            except Exception as e:
                logger.error(f"Failed to send bytes to {user_id}: {e}")
                self.disconnect(user_id)

    async def send_status(self, user_id: str, state: str):
        """Send status update (listening/thinking/speaking)."""
        await self.send_json(user_id, {"type": "status", "state": state})


# Global singleton
manager = ConnectionManager()
