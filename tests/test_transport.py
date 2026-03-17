"""Tests for WebSocket connection manager."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from server.transport import ConnectionManager, VoiceSession


class TestConnectionManager:
    def setup_method(self):
        self.mgr = ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_creates_session(self):
        ws = AsyncMock()
        session = await self.mgr.connect(ws, "user-1")
        assert session.user_id == "user-1"
        assert session.connected is True
        assert self.mgr.active_connections == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes_session(self):
        ws = AsyncMock()
        await self.mgr.connect(ws, "user-1")
        self.mgr.disconnect("user-1")
        assert self.mgr.active_connections == 0
        assert not self.mgr.is_connected("user-1")

    @pytest.mark.asyncio
    async def test_single_session_per_user(self):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await self.mgr.connect(ws1, "user-1")
        await self.mgr.connect(ws2, "user-1")
        # Should still be 1 connection (old one replaced)
        assert self.mgr.active_connections == 1
        # Old websocket should have been closed
        ws1.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_connected(self):
        ws = AsyncMock()
        await self.mgr.connect(ws, "user-1")
        assert self.mgr.is_connected("user-1")
        assert not self.mgr.is_connected("user-2")

    @pytest.mark.asyncio
    async def test_send_json(self):
        ws = AsyncMock()
        await self.mgr.connect(ws, "user-1")
        await self.mgr.send_json("user-1", {"type": "status", "state": "listening"})
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_bytes(self):
        ws = AsyncMock()
        await self.mgr.connect(ws, "user-1")
        await self.mgr.send_bytes("user-1", b"audio_data")
        ws.send_bytes.assert_called_once_with(b"audio_data")

    @pytest.mark.asyncio
    async def test_send_status(self):
        ws = AsyncMock()
        await self.mgr.connect(ws, "user-1")
        await self.mgr.send_status("user-1", "thinking")
        ws.send_json.assert_called_with({"type": "status", "state": "thinking"})

    @pytest.mark.asyncio
    async def test_send_to_disconnected_user_no_error(self):
        await self.mgr.send_json("nonexistent", {"type": "test"})
        # Should not raise

    def test_get_session(self):
        assert self.mgr.get_session("user-1") is None


class TestVoiceSession:
    def test_defaults(self):
        ws = MagicMock()
        s = VoiceSession(user_id="u1", websocket=ws)
        assert s.connected is True
        assert s.turn_count == 0
        assert s.conversation_id is None
