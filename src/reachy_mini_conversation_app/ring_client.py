"""Thin wrapper around the unofficial ring_doorbell library for camera snapshots.

Ring has no official public API for personal use; this talks to the same
reverse-engineered backend as the Ring mobile app via ``ring_doorbell``.
"""

from __future__ import annotations
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ring_doorbell import Auth, Ring, RingError, AuthenticationError
from ring_doorbell.doorbot import RingDoorBell


logger = logging.getLogger(__name__)

RING_USER_AGENT = "reachy-mini-conversation-app"
RING_TOKEN_CACHE_FILENAME = "ring_token.v1.json"
RING_TOKEN_CACHE_PATH_ENV = "RING_TOKEN_CACHE_PATH"

# ring_doorbell's bundled `async_get_snapshot` polls a legacy Ring endpoint that
# no longer reliably serves images (see
# python-ring-doorbell/python-ring-doorbell#527); it now raises IndexError or
# silently returns None for most devices, regardless of device settings. We
# instead call the newer on-demand endpoint used by the official Ring app and
# by ring-mqtt directly, the same fix proposed upstream in PR #528 but not yet
# released. `after-ms`/`max-wait-ms`/`extras=force` ask Ring to generate a
# fresh image on the fly if none younger than `max_age` seconds already exists.
_SNAPSHOT_API_URI = "https://app-snaps.ring.com"
_SNAPSHOT_ENDPOINT = "/snapshots/next/{0}"
_SNAPSHOT_MAX_AGE_S = 30
_SNAPSHOT_MAX_WAIT_S = 10
_SNAPSHOT_ATTEMPTS = 3
_SNAPSHOT_ATTEMPT_BACKOFF_S = 2

# Norwegian synonyms for the English device names recommended in the README, so the
# tool matches the same device no matter which language the user asks in.
LOCATION_ALIASES: dict[str, str] = {
    "hage": "garden",
    "hagen": "garden",
    "framsiden": "front door",
}


class RingNotConfiguredError(Exception):
    """Raised when no cached Ring login is available yet."""


class RingDeviceNotFoundError(Exception):
    """Raised when no Ring device matches the requested location."""


@dataclass(frozen=True)
class RingEvent:
    """A single motion/ding/on_demand entry from a Ring device's event history."""

    device_name: str
    event_id: int
    kind: str
    created_at: datetime


# How many recent history entries to scan per device when looking for the latest
# event of a given kind — Ring's history is newest-first, so this only needs to be
# large enough to skip past a few unrelated kinds (e.g. on_demand) between events.
_HISTORY_LOOKBACK = 10


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


async def _take_snapshot(ring: Ring, device: RingDoorBell) -> bytes:
    """Fetch one snapshot from Ring's on-demand endpoint, forcing a fresh capture."""
    params = {
        "after-ms": int((time.time() - _SNAPSHOT_MAX_AGE_S) * 1000),
        "max-wait-ms": _SNAPSHOT_MAX_WAIT_S * 1000,
        "extras": "force",
    }
    response = await ring.async_query(
        _SNAPSHOT_ENDPOINT.format(device.id),
        extra_params=params,
        base_uri=_SNAPSHOT_API_URI,
        timeout=_SNAPSHOT_MAX_WAIT_S + 5,
    )
    if not response.content:
        raise RuntimeError(f"Ring device '{device.name}' did not return a snapshot")
    return response.content


async def _get_snapshot_with_retry(ring: Ring, device: RingDoorBell) -> bytes:
    """Fetch a snapshot from a Ring device, retrying transient server timeouts.

    A 404/timeout from the on-demand endpoint typically just means the device
    hasn't produced a fresh image within `max_wait`, which is worth retrying
    rather than treating as a permanent failure.
    """
    last_error: Exception | None = None
    for attempt in range(_SNAPSHOT_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_SNAPSHOT_ATTEMPT_BACKOFF_S)
        try:
            return await _take_snapshot(ring, device)
        except (RingError, RuntimeError) as e:
            last_error = e

    raise RuntimeError(
        f"Ring device '{device.name}' did not return a snapshot after {_SNAPSHOT_ATTEMPTS} attempts. "
        "It may be asleep or offline."
    ) from last_error


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

        return await _get_snapshot_with_retry(ring, matching_device)

    async def async_get_latest_events(self, kinds: tuple[str, ...]) -> dict[str, RingEvent]:
        """Return each device's most recent history entry whose kind is in `kinds`.

        Devices with no matching history are omitted from the result. A device
        that fails to return history (e.g. a transient Ring API error) is
        skipped rather than failing the whole call, so one flaky camera can't
        block checking the others.
        """
        ring = await self._get_ring()
        await ring.async_update_devices()

        events: dict[str, RingEvent] = {}
        for device in ring.video_devices():
            try:
                history = await device.async_history(limit=_HISTORY_LOOKBACK)
            except (RingError, RuntimeError) as e:
                logger.warning("Failed to fetch Ring history for '%s': %s", device.name, e)
                continue

            latest = next((entry for entry in history if entry.get("kind") in kinds), None)
            if latest is None:
                continue

            events[device.name] = RingEvent(
                device_name=device.name,
                event_id=latest["id"],
                kind=latest["kind"],
                created_at=latest["created_at"],
            )
        return events

    async def async_list_locations(self) -> list[str]:
        """Return the friendly names of all doorbell/camera devices on the account."""
        ring = await self._get_ring()
        await ring.async_update_devices()
        return [device.name for device in ring.video_devices()]

    async def async_close(self) -> None:
        """Close the underlying HTTP session, if one was ever created."""
        if self._ring is not None:
            await self._ring.auth.async_close()
            self._ring = None
