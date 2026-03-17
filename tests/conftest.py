"""Pytest configuration and shared fixtures."""

import pytest
import os

# Ensure test environment doesn't use real API keys
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("DEEPGRAM_API_KEY", "")
os.environ.setdefault("CARTESIA_API_KEY", "")
os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
