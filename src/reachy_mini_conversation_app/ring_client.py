"""Thin wrapper around the unofficial ring_doorbell library for camera snapshots.

Ring has no official public API for personal use; this talks to the same
reverse-engineered backend as the Ring mobile app via ``ring_doorbell``.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path

from ring_doorbell import Auth, Ring, AuthenticationError


logger = logging.getLogger(__name__)

RING_USER_AGENT = "reachy-mini-conversation-app"
RING_TOKEN_CACHE_FILENAME = "ring_token.v1.json"
RING_TOKEN_CACHE_PATH_ENV = "RING_TOKEN_CACHE_PATH"

# Norwegian synonyms for the English device names recommended in the README, so the
# tool matches the same device no matter which language the user asks in.
LOCATION_ALIASES: dict[str, str] = {
    "hage": "garden",
    "framsiden": "front door",
}


class RingNotConfiguredError(Exception):
    """Raised when no cached Ring login is available yet."""


class RingDeviceNotFoundError(Exception):
    """Raised when no Ring device matches the requested location."""


def token_cache_path(instance_path: str | Path | None = None) -> Path:
    """Return the path used to persist the cached Ring OAuth token."""
    override = os.getenv(RING_TOKEN_CACHE_PATH_ENV)
    if override:
        return Path(override).expanduser()

    if instance_path is not None:
        return Path(instance_path).expanduser() / RING_TOKEN_CACHE_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "reachy_mini_conversation_app" / RING_TOKEN_CACHE_FILENAME


def write_token_cache(path: Path, token: dict[str, object]) -> None:
    """Persist the Ring OAuth token, restricting the file to owner read/write only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token))
    os.chmod(path, 0o600)


class RingClient:
    """Fetches snapshots from the user's Ring devices by friendly location name."""

    def __init__(self, instance_path: str | Path | None = None) -> None:
        """Store the token cache path; the Ring session is created lazily."""
        self._token_cache_path = token_cache_path(instance_path)
        self._ring: Ring | None = None

    def _save_token(self, token: dict[str, object]) -> None:
        write_token_cache(self._token_cache_path, token)

    async def _get_ring(self) -> Ring:
        """Return a Ring session created from the cached token, refreshing as needed."""
        if self._ring is not None:
            return self._ring

        if not self._token_cache_path.is_file():
            raise RingNotConfiguredError(
                "No cached Ring login found. Run `reachy-mini-conversation-app --ring-login` first."
            )

        cached_token = json.loads(self._token_cache_path.read_text())
        auth = Auth(RING_USER_AGENT, cached_token, self._save_token)
        ring = Ring(auth)
        try:
            await ring.async_create_session()
        except AuthenticationError as e:
            raise RingNotConfiguredError(
                "Cached Ring login has expired. Run `reachy-mini-conversation-app --ring-login` again."
            ) from e

        self._ring = ring
        return ring

    async def async_get_device_snapshot(self, location: str) -> bytes:
        """Fetch a fresh JPEG snapshot from the Ring device matching `location` by name."""
        ring = await self._get_ring()
        await ring.async_update_devices()

        video_devices = ring.video_devices()
        normalized_location = location.strip().casefold()
        normalized_location = LOCATION_ALIASES.get(normalized_location, normalized_location)
        matching_device = next(
            (device for device in video_devices if device.name.strip().casefold() == normalized_location),
            None,
        )
        if matching_device is None:
            known_names = ", ".join(sorted(device.name for device in video_devices))
            raise RingDeviceNotFoundError(f"No Ring device named '{location}'. Known devices: {known_names}")

        snapshot = await matching_device.async_get_snapshot()
        if snapshot is None:
            raise RuntimeError(f"Ring device '{matching_device.name}' did not return a snapshot")
        return snapshot

    async def async_list_locations(self) -> list[str]:
        """Return the friendly names of all doorbell/camera devices on the account."""
        ring = await self._get_ring()
        await ring.async_update_devices()
        return [device.name for device in ring.video_devices()]
