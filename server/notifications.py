"""Push notifications via Apple Push Notification service (APNs)."""

from __future__ import annotations
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — aioapns may not be needed if not configured
_apns_client = None


def _get_apns_client():
    """Lazily initialize the APNs client."""
    global _apns_client
    if _apns_client is not None:
        return _apns_client

    key_id = os.environ.get("APNS_KEY_ID", "")
    team_id = os.environ.get("APNS_TEAM_ID", "")
    key_contents = os.environ.get("APNS_KEY_CONTENTS", "")

    if not all([key_id, team_id, key_contents]):
        logger.info("APNs not configured — push notifications disabled")
        return None

    try:
        from aioapns import APNs, NotificationRequest

        # aioapns needs a file path — write key contents to a temp file
        key_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".p8", delete=False, prefix="apns_key_"
        )
        key_file.write(key_contents)
        key_file.close()

        _apns_client = APNs(
            key=key_file.name,
            key_id=key_id,
            team_id=team_id,
            topic="com.evernear.app",  # iOS app bundle ID
            use_sandbox=True,  # Switch to False for production
        )
        logger.info(f"APNs client initialized (key_id={key_id}, team_id={team_id}, sandbox=True)")
        return _apns_client
    except Exception as e:
        logger.error(f"Failed to initialize APNs client: {e}")
        return None


class PushNotificationService:
    """Apple Push Notification service wrapper."""

    async def send_medication_reminder(
        self,
        device_token: str,
        preferred_name: str,
    ) -> bool:
        """Send a warm medication reminder notification."""
        return await self.send_notification(
            device_token=device_token,
            title="EverNear",
            body=f"Hi {preferred_name}, just a friendly reminder about your medication.",
            category="medication_reminder",
        )

    async def send_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        category: str = "",
    ) -> bool:
        """Send a push notification via APNs."""
        client = _get_apns_client()
        if client is None:
            logger.info(f"[STUB] Would send notification: {title} — {body}")
            return False

        try:
            from aioapns import NotificationRequest

            request = NotificationRequest(
                device_token=device_token,
                message={
                    "aps": {
                        "alert": {
                            "title": title,
                            "body": body,
                        },
                        "sound": "default",
                        **({"category": category} if category else {}),
                    }
                },
            )

            response = await client.send_notification(request)

            if response.is_successful:
                logger.info(f"APNs notification sent: {title}")
                return True
            else:
                logger.error(f"APNs send failed: {response.description}")
                return False

        except Exception as e:
            logger.error(f"APNs send error: {e}")
            return False


# Global singleton
push_service = PushNotificationService()
