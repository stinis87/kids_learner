"""Ring event watcher: notice motion/doorbell events and speak up on its own.

Used by profiles that opt in via a ``ring_watcher.txt`` marker file in their
profile directory (see :func:`load_config`). Polls each Ring device's event
history on an interval and, when a genuinely new motion or doorbell event
appears, fetches a fresh snapshot and prompts the model to react — the same
"inject an image, let the model respond freely" pattern used by
:class:`reachy_mini_conversation_app.proactive_vision.ProactiveVisionEngine`.
"""

from __future__ import annotations
import time
import base64
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from reachy_mini_conversation_app.ring_client import RingClient


logger = logging.getLogger(__name__)

CONFIG_FILENAME = "ring_watcher.txt"

# Which Ring history "kind" values are worth reacting to. "on_demand" (someone
# opened a live view/snapshot in the Ring app) is deliberately excluded.
_WATCHED_KINDS = ("motion", "ding")

_NUDGE_TEMPLATES = {
    "ding": "(Someone just rang the doorbell at {device} — look and react.)",
    "motion": "(Motion was just detected at {device} — look and react.)",
}

# A physical reaction to play immediately when a new event comes in, before the
# model has even started forming a spoken response — makes the reaction feel
# instantaneous instead of only speaking a beat later.
_EMOTION_FOR_KIND = {
    "ding": "excited",
    "motion": "attentive",
}


@dataclass(frozen=True)
class RingWatcherConfig:
    """Timing knobs for the Ring watcher loop, read from a profile marker file."""

    poll_interval_seconds: float = 20.0
    device_cooldown_seconds: float = 120.0


def load_config(profile_dir: Path) -> RingWatcherConfig | None:
    """Return the profile's Ring watcher config, or None when it opts out."""
    config_file = profile_dir / CONFIG_FILENAME
    if not config_file.exists():
        return None

    values: dict[str, float] = {}
    try:
        for line in config_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, raw_value = stripped.partition("=")
            try:
                values[key.strip()] = float(raw_value.strip())
            except ValueError:
                logger.warning("Ignoring malformed %s line: %r", CONFIG_FILENAME, line)
    except Exception as e:
        logger.warning("Failed to read %s: %s", config_file, e)
        return RingWatcherConfig()

    defaults = RingWatcherConfig()
    return RingWatcherConfig(
        poll_interval_seconds=values.get("poll_interval_seconds", defaults.poll_interval_seconds),
        device_cooldown_seconds=values.get("device_cooldown_seconds", defaults.device_cooldown_seconds),
    )


class RingWatcherEngine:
    """Polls Ring devices for new events and prompts the model to react on its own."""

    def __init__(
        self,
        config: RingWatcherConfig,
        *,
        ring_client: RingClient,
        is_ready: Callable[[], bool],
        send_prompt: Callable[[str, str], Awaitable[None]],
        play_emotion: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Store collaborators; nothing runs until :meth:`start`."""
        self._config = config
        self._ring_client = ring_client
        self._is_ready = is_ready
        self._send_prompt = send_prompt
        self._play_emotion = play_emotion
        self._task: asyncio.Task[None] | None = None
        self._last_event_ids: dict[str, int] = {}
        self._last_reacted_at: dict[str, float] = {}
        self._seeded = False

    def start(self) -> None:
        """Start the background polling loop if it isn't already running."""
        if self._task is not None and not self._task.done():
            return
        self._last_event_ids = {}
        self._last_reacted_at = {}
        self._seeded = False
        self._task = asyncio.create_task(self._run(), name="ring-watcher")

    async def stop(self) -> None:
        """Stop the background polling loop."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._config.poll_interval_seconds)
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Ring watcher poll failed: %s", e)

    async def _poll_once(self) -> None:
        events = await self._ring_client.async_get_latest_events(_WATCHED_KINDS)

        if not self._seeded:
            # Seed "last seen" ids from whatever is already in history so we don't
            # react to stale events from before the watcher started.
            self._last_event_ids = {name: event.event_id for name, event in events.items()}
            self._seeded = True
            return

        for device_name, event in events.items():
            if self._last_event_ids.get(device_name) == event.event_id:
                continue
            self._last_event_ids[device_name] = event.event_id

            last_reacted_at = self._last_reacted_at.get(device_name, 0.0)
            if time.monotonic() - last_reacted_at < self._config.device_cooldown_seconds:
                continue
            if not self._is_ready():
                continue

            try:
                jpeg_bytes = await self._ring_client.async_get_device_snapshot(device_name)
            except Exception as e:
                logger.warning("Ring watcher could not fetch a snapshot for '%s': %s", device_name, e)
                continue

            self._last_reacted_at[device_name] = time.monotonic()

            if self._play_emotion is not None:
                emotion = _EMOTION_FOR_KIND.get(event.kind)
                if emotion is not None:
                    try:
                        await self._play_emotion(emotion)
                    except Exception as e:
                        logger.warning("Ring watcher could not play emotion '%s': %s", emotion, e)

            nudge_template = _NUDGE_TEMPLATES.get(event.kind, "(Something happened at {device} — look and react.)")
            b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
            await self._send_prompt(b64_image, nudge_template.format(device=device_name))
