"""Tests for Auth — JWT validation."""

import pytest
import jwt as pyjwt
import time
from unittest.mock import patch
from server.auth import decode_jwt, get_user_id, authenticate_websocket


class TestDecodeJWT:
    def _make_token(self, sub: str = "user-123", exp: int | None = None) -> str:
        payload = {"sub": sub, "iss": "supabase", "role": "authenticated"}
        if exp:
            payload["exp"] = exp
        return pyjwt.encode(payload, "secret", algorithm="HS256")

    def test_valid_token_returns_payload(self):
        token = self._make_token(sub="user-456")
        payload = decode_jwt(token)
        assert payload["sub"] == "user-456"

    def test_missing_sub_raises(self):
        token = pyjwt.encode({"iss": "supabase"}, "secret", algorithm="HS256")
        with pytest.raises(Exception):
            decode_jwt(token)

    def test_garbage_token_raises(self):
        with pytest.raises(Exception):
            decode_jwt("not.a.valid.token")


class TestGetUserId:
    def test_extracts_sub(self):
        token = pyjwt.encode({"sub": "user-789"}, "secret", algorithm="HS256")
        assert get_user_id(token) == "user-789"


class TestAuthenticateWebSocket:
    @pytest.mark.asyncio
    async def test_no_token_returns_none(self):
        from unittest.mock import MagicMock
        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        result = await authenticate_websocket(ws)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_header_token(self):
        from unittest.mock import MagicMock
        token = pyjwt.encode({"sub": "user-ws"}, "secret", algorithm="HS256")
        ws = MagicMock()
        ws.headers = {"authorization": f"Bearer {token}"}
        ws.query_params = {}
        result = await authenticate_websocket(ws)
        assert result == "user-ws"

    @pytest.mark.asyncio
    async def test_query_param_token(self):
        from unittest.mock import MagicMock
        token = pyjwt.encode({"sub": "user-qp"}, "secret", algorithm="HS256")
        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {"token": token}
        result = await authenticate_websocket(ws)
        assert result == "user-qp"

    @pytest.mark.asyncio
    async def test_invalid_token_returns_none(self):
        from unittest.mock import MagicMock
        ws = MagicMock()
        ws.headers = {"authorization": "Bearer garbage"}
        ws.query_params = {}
        result = await authenticate_websocket(ws)
        assert result is None
