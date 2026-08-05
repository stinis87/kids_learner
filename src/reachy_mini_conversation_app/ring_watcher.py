"""Ring event watcher: notice motion/doorbell events and speak up on its own.

Enabled by default for every profile whenever Ring is configured (the Ring
account is a single shared credential, not per-personality data). A profile
can tune the cooldown, or opt out entirely, via an optional ``ring_watcher.txt``
file in its profile directory (see :func:`load_config`). Subscribes to Ring's
push notification stream (the same FCM mechanism the Ring app itself uses) and,
when a genuinely new motion or doorbell event arrives, fetches a fresh snapshot
and prompts the model to react — the same "inject an image, let the model
respond freely" pattern used by
:class:`reachy_mini_conversation_app.proactive_vision.ProactiveVisionEngine`.

A profile can additionally set ``door_call_mode`` to change how a doorbell
"ding" is handled: ``ask`` (the default) has Reachy ask out loud whether it
should answer itself or let the person nearby talk, ``auto`` opens a live
two-way audio call (see :mod:`reachy_mini_conversation_app.ring_call`)
straight away with no question asked, and ``off`` keeps the original
snapshot-only nudge with no call offered.
"""

from __future__ import annotations
import time
import base64
import asyncio
import logging
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from reachy_mini_conversation_app.ring_client import RingEvent, RingClient


logger = logging.getLogger(__name__)

CONFIG_FILENAME = "ring_watcher.txt"

_NUDGE_TEMPLATES = {
    "ding": "(Someone just rang the doorbell at {device} — look and react.)",
    "motion": "(Motion was just detected at {device} — look and react.)",
}

# Sent instead of the plain "ding" nudge above when door_call_mode="ask": asks
# the user out loud and lets the model act on the answer with the existing
# talk_to_door/end_door_call tools, rather than deciding anything on its own.
_DOOR_CALL_ASK_TEMPLATE = (
    "(Someone just rang the doorbell at {device}. Ask out loud whether you should answer the "
    "door yourself or whether they'd rather talk to the visitor themselves. If they want you to "
    'handle it, call talk_to_door with location="{device}" and carry the conversation with the '
    "visitor on your own. If they'd rather talk themselves, call talk_to_door with the same "
    "location so their voice reaches the visitor, then let the conversation follow their lead. "
    "If they say to ignore it, don't open a call. Call end_door_call once it's over.)"
)

DOOR_CALL_MODES = ("off", "ask", "auto")

# A physical reaction to play immediately when a new event comes in, before the
# model has even started forming a spoken response — makes the reaction feel
# instantaneous instead of only speaking a beat later.
_EMOTION_FOR_KIND = {
    "ding": "excited",
    "motion": "attentive",
}

# How often to log a heartbeat confirming the push listener is still connected,
# even when nothing has happened — lets you tell "silently idle" apart from "the
# push connection died" without flooding the log on every notification.
_HEARTBEAT_INTERVAL_SECONDS = 900.0


@dataclass(frozen=True)
class RingWatcherConfig:
    """Timing knobs for the Ring watcher, read from a profile marker file."""

    device_cooldown_seconds: float = 120.0
    door_call_mode: Literal["off", "ask", "auto"] = "ask"


def load_config(profile_dir: Path) -> RingWatcherConfig | None:
    """Return the profile's Ring watcher config; enabled by default, None when opted out."""
    config_file = profile_dir / CONFIG_FILENAME
    if not config_file.exists():
        return RingWatcherConfig()

    enabled = True
    door_call_mode = RingWatcherConfig().door_call_mode
    values: dict[str, float] = {}
    try:
        for line in config_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, raw_value = stripped.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            if key == "enabled":
                enabled = raw_value.lower() not in ("false", "0", "no")
                continue
            if key == "door_call_mode":
                normalized = raw_value.lower()
                if normalized in DOOR_CALL_MODES:
                    door_call_mode = normalized  # type: ignore[assignment]
                else:
                    logger.warning("Ignoring unknown door_call_mode value: %r", raw_value)
                continue
            try:
                values[key] = float(raw_value)
            except ValueError:
                logger.warning("Ignoring malformed %s line: %r", CONFIG_FILENAME, line)
    except Exception as e:
        logger.warning("Failed to read %s: %s", config_file, e)
        return RingWatcherConfig()

    if not enabled:
        return None

    defaults = RingWatcherConfig()
    return RingWatcherConfig(
        device_cooldown_seconds=values.get("device_cooldown_seconds", defaults.device_cooldown_seconds),
        door_call_mode=door_call_mode,
    )


class RingWatcherEngine:
    """Listens for Ring push notifications and prompts the model to react on its own."""

    def __init__(
        self,
        config: RingWatcherConfig,
        *,
        ring_client: RingClient,
        is_ready: Callable[[], bool],
        send_prompt: Callable[[str, str], Awaitable[None]],
        play_emotion: Callable[[str], Awaitable[None]] | None = None,
        answer_ding: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Store collaborators; nothing runs until :meth:`start`.

        `answer_ding` is called instead of `send_prompt` for "ding" events when
        `config.door_call_mode == "auto"` — it's expected to open a live doorbell
        call rather than just nudge the model with a snapshot.
        """
        self._config = config
        self._ring_client = ring_client
        self._is_ready = is_ready
        self._send_prompt = send_prompt
        self._play_emotion = play_emotion
        self._answer_ding = answer_ding
        self._listen_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._last_reacted_at: dict[str, float] = {}

    def start(self) -> None:
        """Start the push notification listener if it isn't already running."""
        if self._listen_task is not None and not self._listen_task.done():
            return
        self._last_reacted_at = {}
        self._listen_task = asyncio.create_task(self._listen(), name="ring-watcher-listen")
        self._health_task = asyncio.create_task(self._health_check_loop(), name="ring-watcher-health")

    async def stop(self) -> None:
        """Stop the push notification listener."""
        await self._ring_client.async_stop_event_listener()
        for task in (self._listen_task, self._health_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._listen_task = None
        self._health_task = None

    async def _listen(self) -> None:
        started = await self._ring_client.async_start_event_listener(self._on_push_event)
        if started:
            logger.info("Ring watcher: listening for Ring push notifications")
        else:
            logger.warning("Ring watcher: push listener failed to start, Ring events will not be reacted to")

    async def _health_check_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
            if self._ring_client.is_event_listener_active():
                logger.info("Ring watcher: still listening for push notifications")
            else:
                logger.warning("Ring watcher: push listener is no longer active")

    def _on_push_event(self, event: RingEvent) -> None:
        """Schedule a reaction to a new push notification from the FCM push client."""
        asyncio.create_task(self._react_to_event(event))

    async def _react_to_event(self, event: RingEvent) -> None:
        device_name = event.device_name

        # A doorbell press is a deliberate, distinct action from a visitor — never
        # suppress it behind an unrelated cooldown from a recent motion reaction
        # (motion almost always precedes a ding as someone walks up to the door).
        last_reacted_at = self._last_reacted_at.get(device_name, 0.0)
        if event.kind != "ding" and time.monotonic() - last_reacted_at < self._config.device_cooldown_seconds:
            logger.debug(
                "Ring watcher: new '%s' event at '%s' (id=%s) ignored, still in cooldown",
                event.kind,
                device_name,
                event.event_id,
            )
            return
        if not self._is_ready():
            logger.debug(
                "Ring watcher: new '%s' event at '%s' (id=%s) ignored, model not ready",
                event.kind,
                device_name,
                event.event_id,
            )
            return

        logger.info(
            "Ring watcher: reacting to new '%s' event at '%s' (id=%s)", event.kind, device_name, event.event_id
        )

        self._last_reacted_at[device_name] = time.monotonic()

        if self._play_emotion is not None:
            emotion = _EMOTION_FOR_KIND.get(event.kind)
            if emotion is not None:
                try:
                    await self._play_emotion(emotion)
                except Exception as e:
                    logger.warning("Ring watcher could not play emotion '%s': %s", emotion, e)

        if event.kind == "ding" and self._config.door_call_mode == "auto" and self._answer_ding is not None:
            try:
                await self._answer_ding(device_name)
            except Exception as e:
                logger.warning("Ring watcher could not answer the doorbell at '%s': %s", device_name, e)
            return

        try:
            jpeg_bytes = await self._ring_client.async_get_device_snapshot(device_name)
        except Exception as e:
            logger.warning("Ring watcher could not fetch a snapshot for '%s': %s", device_name, e)
            return

        if event.kind == "ding" and self._config.door_call_mode == "ask":
            nudge_text = _DOOR_CALL_ASK_TEMPLATE.format(device=device_name)
        else:
            nudge_template = _NUDGE_TEMPLATES.get(event.kind, "(Something happened at {device} — look and react.)")
            nudge_text = nudge_template.format(device=device_name)

        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        await self._send_prompt(b64_image, nudge_text)
