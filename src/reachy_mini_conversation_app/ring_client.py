"""Thin wrapper around the unofficial ring_doorbell library for camera snapshots.

Ring has no official public API for personal use; this talks to the same
reverse-engineered backend as the Ring mobile app via ``ring_doorbell``.
"""

from __future__ import annotations
import os
import re
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from collections.abc import Sequence

from ring_doorbell import Auth, Ring, RingError, AuthenticationError
from ring_doorbell.doorbot import RingDoorBell

from reachy_mini_conversation_app.video_frames import FfmpegNotAvailableError, async_extract_evenly_spaced_frames


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


class RingDayNotRecognizedError(Exception):
    """Raised when a requested day string can't be resolved to a calendar date."""


class RingNoEventsFoundError(Exception):
    """Raised when a device has no matching history entries for the requested day."""


class RingRecordingUnavailableError(Exception):
    """Raised when a recording can't be downloaded (no Ring Protect, or Ring error)."""


class RingEventNotFoundError(Exception):
    """Raised when an event selector doesn't match any event in the requested day."""


@dataclass(frozen=True)
class RingEvent:
    """A single motion/ding/on_demand entry from a Ring device's event history."""

    device_name: str
    event_id: int
    kind: str
    created_at: datetime


@dataclass(frozen=True)
class RingHistorySummary:
    """A device's motion/ding events for one calendar day, newest first."""

    device_name: str
    day: date
    events: list[RingEvent]


# Which Ring history "kind" values are worth surfacing to the user. "on_demand"
# (someone opened a live view/snapshot in the Ring app) is deliberately excluded.
# Shared with ring_watcher.py so both the proactive watcher and this retroactive
# history query agree on what counts as an "event".
WATCHED_HISTORY_KINDS = ("motion", "ding")

# How many recent history entries to scan per device when looking for the latest
# event of a given kind — Ring's history is newest-first, so this only needs to be
# large enough to skip past a few unrelated kinds (e.g. on_demand) between events.
_HISTORY_LOOKBACK = 10

# Paging size/cap when scanning a full day of history for `async_get_history_for_day`.
# Ring's `older_than` cursor pages by event id with no native date filter, so we page
# newest-first until an entry falls before the requested day, capped so a Ring
# account with unusually dense history can't turn one query into unbounded requests.
_HISTORY_DAY_PAGE_SIZE = 50
_HISTORY_DAY_MAX_PAGES = 20

# Recorded clips need an active Ring Protect subscription and can take a while to
# transfer; a longer timeout than the on-demand snapshot endpoint's.
_RECORDING_DOWNLOAD_TIMEOUT_S = 60

# Frames to sample across a described clip's duration — enough to catch a subject
# who doesn't stay in one spot, without the cost of decoding every frame.
_DESCRIBE_FRAME_COUNT = 4

_DAY_ALIASES = {
    "today": 0,
    "yesterday": 1,
    "day before yesterday": 2,
    "the day before yesterday": 2,
    "day_before_yesterday": 2,
    # Norwegian, so the tool matches the same day no matter which language the user
    # asks in — mirrors LOCATION_ALIASES' existing Norwegian device-name synonyms.
    "i dag": 0,
    "idag": 0,
    "i går": 1,
    "igår": 1,
    "i gaar": 1,
    "i forgårs": 2,
    "iforgårs": 2,
    "i forgaars": 2,
}


def resolve_day(day: str, tz: ZoneInfo) -> date:
    """Resolve `day` ('today', 'yesterday', 'day before yesterday', or 'YYYY-MM-DD') to a date."""
    normalized = day.strip().casefold()
    days_ago = _DAY_ALIASES.get(normalized)
    if days_ago is not None:
        return (datetime.now(tz) - timedelta(days=days_ago)).date()

    try:
        return date.fromisoformat(normalized)
    except ValueError:
        raise RingDayNotRecognizedError(
            f"Could not understand day '{day}'. Use 'today', 'yesterday', 'day before yesterday', or YYYY-MM-DD."
        ) from None


# Selectors naming the most/least recent event, in English and Norwegian.
_LATEST_EVENT_ALIASES = {"latest", "most recent", "last", "newest", "siste", "nyeste"}

# Ordinal position counted chronologically from the day's earliest event (index 0),
# so "first"/"1st"/"1" and "earliest"/"oldest" all mean the same thing. English and
# Norwegian ordinals, matching the day-name aliases' bilingual convention above.
_ORDINAL_EVENT_ALIASES = {
    "first": 0,
    "earliest": 0,
    "oldest": 0,
    "1st": 0,
    "1": 0,
    "første": 0,
    "eldste": 0,
    "second": 1,
    "2nd": 1,
    "2": 1,
    "andre": 1,
    "third": 2,
    "3rd": 2,
    "3": 2,
    "tredje": 2,
    "fourth": 3,
    "4th": 3,
    "4": 3,
    "fjerde": 3,
    "fifth": 4,
    "5th": 4,
    "5": 4,
    "femte": 4,
}

# Matches a clock time like "14:00", "2pm", "2:30 pm", or the Norwegian "kl. 14".
_TIME_OF_DAY_PATTERN = re.compile(r"(?:kl\.?\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE)


def _parse_time_of_day(selector: str) -> clock_time | None:
    """Parse a clock-time selector (e.g. '14:00', '2pm') into a `time`, or None if it isn't one."""
    match = _TIME_OF_DAY_PATTERN.fullmatch(selector.strip())
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return clock_time(hour, minute)


def _select_event(events: list[RingEvent], selector: str) -> RingEvent:
    """Pick one event from a day's list (newest-first) by name, ordinal, or clock time.

    `events` must be non-empty; callers check that first so a "no events at all"
    day gets its own, more specific error.
    """
    normalized = selector.strip().casefold() if selector else "latest"
    if not normalized or normalized in _LATEST_EVENT_ALIASES:
        return events[0]

    ordinal = _ORDINAL_EVENT_ALIASES.get(normalized)
    if ordinal is not None:
        # `events` is newest-first; ordinal 0 means the day's earliest event, at the end.
        index_from_latest = len(events) - 1 - ordinal
        if index_from_latest < 0:
            raise RingEventNotFoundError(f"There were only {len(events)} event(s) that day, no '{selector}' one.")
        return events[index_from_latest]

    target_time = _parse_time_of_day(normalized)
    if target_time is not None:
        target_seconds = target_time.hour * 3600 + target_time.minute * 60
        return min(
            events,
            key=lambda event: abs(
                event.created_at.hour * 3600 + event.created_at.minute * 60 + event.created_at.second - target_seconds
            ),
        )

    raise RingEventNotFoundError(f"Could not understand which event '{selector}' refers to.")


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


def _match_device(video_devices: Sequence[RingDoorBell], location: str) -> RingDoorBell:
    """Return the video device whose name matches `location`, applying `LOCATION_ALIASES`."""
    normalized_location = location.strip().casefold()
    normalized_location = LOCATION_ALIASES.get(normalized_location, normalized_location)
    matching_device = next(
        (device for device in video_devices if device.name.strip().casefold() == normalized_location),
        None,
    )
    if matching_device is None:
        known_names = ", ".join(sorted(device.name for device in video_devices))
        raise RingDeviceNotFoundError(f"No Ring device named '{location}'. Known devices: {known_names}")
    return matching_device


async def _fetch_day_events(device: RingDoorBell, window_start: datetime, window_end: datetime) -> list[RingEvent]:
    """Page a device's history newest-first, collecting watched-kind events within `[window_start, window_end)`."""
    events: list[RingEvent] = []
    older_than: int | None = None
    for _ in range(_HISTORY_DAY_MAX_PAGES):
        page = await device.async_history(
            limit=_HISTORY_DAY_PAGE_SIZE,
            older_than=older_than,
            timezone=device.timezone,
        )
        if not page:
            break

        reached_window_start = False
        for entry in page:
            created_at = entry["created_at"]
            if created_at >= window_end:
                continue
            if created_at < window_start:
                reached_window_start = True
                break
            if entry.get("kind") in WATCHED_HISTORY_KINDS:
                events.append(
                    RingEvent(
                        device_name=device.name,
                        event_id=entry["id"],
                        kind=entry["kind"],
                        created_at=created_at,
                    )
                )

        if reached_window_start or len(page) < _HISTORY_DAY_PAGE_SIZE:
            break
        older_than = page[-1]["id"]

    return events


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
        matching_device = _match_device(ring.video_devices(), location)
        return await _get_snapshot_with_retry(ring, matching_device)

    async def async_get_history_for_day(self, location: str, day: str) -> RingHistorySummary:
        """Return the device's motion/ding events for `day` ('today', 'yesterday', or 'YYYY-MM-DD')."""
        ring = await self._get_ring()
        await ring.async_update_devices()
        device = _match_device(ring.video_devices(), location)

        tz = ZoneInfo(device.timezone) if device.timezone else ZoneInfo("UTC")
        target_day = resolve_day(day, tz)
        window_start = datetime.combine(target_day, datetime.min.time(), tzinfo=tz)
        window_end = window_start + timedelta(days=1)

        events = await _fetch_day_events(device, window_start, window_end)
        return RingHistorySummary(device_name=device.name, day=target_day, events=events)

    async def async_describe_event(
        self, location: str, day: str, selector: str = "latest"
    ) -> tuple[RingEvent, list[bytes]]:
        """Return one of the day's events (picked by `selector`) and JPEG frames from its recorded clip.

        `selector` accepts 'latest'/'most recent', an ordinal ('first', 'second', ...,
        counted chronologically from the day's earliest event), or a clock time
        ('14:00', '2pm') matched to the closest event — see `_select_event`.

        Requires an active Ring Protect subscription to download the recording, and
        `ffmpeg` installed to extract frames from it.
        """
        ring = await self._get_ring()
        await ring.async_update_devices()
        device = _match_device(ring.video_devices(), location)

        tz = ZoneInfo(device.timezone) if device.timezone else ZoneInfo("UTC")
        target_day = resolve_day(day, tz)
        window_start = datetime.combine(target_day, datetime.min.time(), tzinfo=tz)
        window_end = window_start + timedelta(days=1)

        events = await _fetch_day_events(device, window_start, window_end)
        if not events:
            raise RingNoEventsFoundError(
                f"No motion or doorbell events for '{device.name}' on {target_day.isoformat()}."
            )

        event = _select_event(events, selector)
        if not device.has_subscription:
            raise RingRecordingUnavailableError(
                f"'{device.name}' has no active Ring Protect subscription, so recorded clips can't be downloaded."
            )

        try:
            video_bytes = await device.async_recording_download(event.event_id, timeout=_RECORDING_DOWNLOAD_TIMEOUT_S)
        except RingError as e:
            raise RingRecordingUnavailableError(f"Could not download the recording for '{device.name}': {e}") from e
        if not video_bytes:
            raise RingRecordingUnavailableError(f"Ring returned no recording for '{device.name}'.")

        try:
            frames = await async_extract_evenly_spaced_frames(video_bytes, _DESCRIBE_FRAME_COUNT)
        except FfmpegNotAvailableError as e:
            raise RingRecordingUnavailableError(str(e)) from e

        return event, frames

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
