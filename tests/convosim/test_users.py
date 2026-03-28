"""Test user management for ConvoSim — create/cleanup Supabase users with CONVOSIM_ prefix."""

from __future__ import annotations
import json, jwt, os, time, uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase credentials — read from env or .env in project root
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env from project root if present."""
    from pathlib import Path
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def _get_service_client():
    """Get a Supabase service-role client."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
            "Export them or put them in .env at project root."
        )
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def make_jwt(user_id: str) -> str:
    """Create a minimal JWT with sub claim (signature not verified by server)."""
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600 * 24,
    }
    return jwt.encode(payload, "convosim-test-secret", algorithm="HS256")


def create_test_user(persona_id: str) -> tuple[str, str]:
    """Create a Supabase auth user for a persona. Returns (user_id, jwt_token).

    If Supabase service key is not available, falls back to a synthetic UUID + unsigned JWT.
    """
    email = f"CONVOSIM_{persona_id}_{uuid.uuid4().hex[:6]}@test.evernear.local"

    if SUPABASE_SERVICE_KEY:
        try:
            client = _get_service_client()
            resp = client.auth.admin.create_user({
                "email": email,
                "password": f"convosim-{uuid.uuid4().hex[:8]}",
                "email_confirm": True,
                "user_metadata": {"convosim": True, "persona": persona_id},
            })
            user_id = str(resp.user.id)
            token = make_jwt(user_id)
            logger.info(f"Created Supabase user {email} → {user_id}")
            return user_id, token
        except Exception as e:
            logger.warning(f"Supabase user creation failed ({e}), using synthetic user")

    # Fallback: synthetic user (works because server doesn't verify JWT signature)
    user_id = str(uuid.uuid4())
    token = make_jwt(user_id)
    logger.info(f"Synthetic test user for {persona_id} → {user_id}")
    return user_id, token


def cleanup_test_users():
    """Remove all CONVOSIM_ test users from Supabase."""
    if not SUPABASE_SERVICE_KEY:
        logger.info("No service key — skipping cleanup (synthetic users only)")
        return 0

    client = _get_service_client()
    deleted = 0
    try:
        # List users and filter by CONVOSIM_ prefix
        page = 1
        while True:
            resp = client.auth.admin.list_users(page=page, per_page=100)
            users = resp if isinstance(resp, list) else (resp.users if hasattr(resp, 'users') else [])
            if not users:
                break
            for u in users:
                email = getattr(u, 'email', '') or ''
                if email.startswith("CONVOSIM_"):
                    try:
                        client.auth.admin.delete_user(str(u.id))
                        deleted += 1
                        logger.info(f"Deleted test user {email}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {email}: {e}")
            if len(users) < 100:
                break
            page += 1
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

    logger.info(f"Cleanup complete: {deleted} users deleted")
    return deleted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_test_users()
    else:
        uid, tok = create_test_user("test")
        print(f"User: {uid}\nToken: {tok[:50]}...")
