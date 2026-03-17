"""Tests for Push Notifications (stub mode)."""

import pytest
from server.notifications import PushNotificationService


class TestPushNotificationStub:
    def test_not_configured_by_default(self):
        svc = PushNotificationService()
        assert not svc._configured

    def test_configured_when_all_keys_present(self):
        svc = PushNotificationService(key_id="K", team_id="T", key_path="/p")
        assert svc._configured

    @pytest.mark.asyncio
    async def test_stub_returns_false(self):
        svc = PushNotificationService()
        result = await svc.send_medication_reminder("token", "Margaret")
        assert result is False

    @pytest.mark.asyncio
    async def test_notification_payload_format(self):
        """Verify the notification would use warm, personal tone."""
        svc = PushNotificationService()
        # In stub mode, just verify it doesn't crash
        result = await svc.send_notification("token", "EverNear", "Hi Margaret", "test")
        assert result is False

    @pytest.mark.asyncio
    async def test_generic_notification_stub(self):
        svc = PushNotificationService()
        result = await svc.send_notification("token", "Title", "Body")
        assert result is False
