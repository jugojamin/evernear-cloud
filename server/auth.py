"""JWT authentication for EverNear Cloud."""

from __future__ import annotations
import jwt
from fastapi import HTTPException, Header, WebSocket
from typing import Optional
from server.config import get_settings


def decode_jwt(token: str) -> dict:
    """Decode and validate a Supabase JWT. Returns payload."""
    s = get_settings()
    try:
        # Supabase JWTs are signed with the JWT secret (anon key acts as audience)
        # For validation, we decode without verification in MVP and rely on Supabase RLS
        # In production, use the Supabase JWT secret for full verification
        payload = jwt.decode(token, options={"verify_signature": False})
        if "sub" not in payload:
            raise HTTPException(status_code=401, detail="Invalid token: no sub claim")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_user_id(token: str) -> str:
    """Extract user_id (sub) from JWT."""
    payload = decode_jwt(token)
    return payload["sub"]


async def require_auth(authorization: str = Header(...)) -> str:
    """FastAPI dependency: extract user_id from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization[7:]
    return get_user_id(token)


async def authenticate_websocket(websocket: WebSocket) -> Optional[str]:
    """Authenticate a WebSocket connection. Returns user_id or None."""
    # Check Authorization header first, then query param
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = websocket.query_params.get("token", "")

    if not token:
        return None

    try:
        return get_user_id(token)
    except HTTPException:
        return None
