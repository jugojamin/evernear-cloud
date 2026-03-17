"""Supabase client singleton."""

from supabase import create_client, Client
from server.config import get_settings

_client: Client | None = None
_service_client: Client | None = None


def get_supabase() -> Client:
    """Get anon-key Supabase client (for RLS-scoped queries)."""
    global _client
    if _client is None:
        s = get_settings()
        _client = create_client(s.supabase_url, s.supabase_anon_key)
    return _client


def get_service_client() -> Client:
    """Get service-role client (bypasses RLS — use for server-side operations only)."""
    global _service_client
    if _service_client is None:
        s = get_settings()
        _service_client = create_client(s.supabase_url, s.supabase_service_key)
    return _service_client


def get_user_client(jwt: str) -> Client:
    """Get a Supabase client scoped to a user's JWT for RLS enforcement."""
    s = get_settings()
    client = create_client(s.supabase_url, s.supabase_anon_key)
    client.auth.set_session(jwt, "")
    return client
