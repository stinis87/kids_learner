"""Proactive vision: watch the camera and speak up on its own.

Used by profiles that opt in via a ``proactive_vision.txt`` marker file in their
profile directory (see :func:`load_config`). Samples the camera frequently to
always have a fresh view, but only injects a picture and prompts the model for a
spoken reaction when the scene visibly changed (e.g. a page turn) or after a short
quiet pause — whichever comes first — so Reachy feels continuously attentive
without turning into a non-stop narrator.
"""

from __future__ import annotations
import time
import base64
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Awaitable


logger = logging.getLogger(__name__)

CONFIG_FILENAME = "proactive_vision.txt"

# JPEG size swings this much between two frames of the same steady scene are just
# sensor noise; a page turn or a new subject changes the compressed size a lot more.
# No frame-decoding dependency is available in this app's runtime, so this size
# delta is a deliberately cheap proxy for "the picture changed", not a pixel diff.
_CHANGE_SIZE_RATIO = 0.20

_NUDGE_ON_CHANGE = "(The page/picture just changed — react to what's in front of you now.)"
_NUDGE_ON_QUIET = "(It's been quiet for a bit — look at the picture and say or ask something about it.)"


@dataclass(frozen=True)
class ProactiveVisionConfig:
    """Timing knobs for the proactive vision loop, read from a profile marker file."""

    sample_interval_seconds: float = 3.0
    quiet_pause_seconds: float = 7.0
    speak_cooldown_seconds: float = 10.0


def load_config(profile_dir: Path) -> ProactiveVisionConfig | None:
    """Return the profile's proactive vision config, or None when it opts out."""
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
        return ProactiveVisionConfig()

    defaults = ProactiveVisionConfig()
    return ProactiveVisionConfig(
        sample_interval_seconds=values.get("sample_interval_seconds", defaults.sample_interval_seconds),
        quiet_pause_seconds=values.get("quiet_pause_seconds", defaults.quiet_pause_seconds),
        speak_cooldown_seconds=values.get("speak_cooldown_seconds", defaults.speak_cooldown_seconds),
    )


class ProactiveVisionEngine:
    """Samples the camera on an interval and prompts the model to react on its own."""

    def __init__(
        self,
        config: ProactiveVisionConfig,
        *,
        get_frame_jpeg: Callable[[], bytes | None],
        is_ready: Callable[[], bool],
        seconds_since_activity: Callable[[], float],
        send_prompt: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """Store collaborators; nothing runs until :meth:`start`."""
        self._config = config
        self._get_frame_jpeg = get_frame_jpeg
        self._is_ready = is_ready
        self._seconds_since_activity = seconds_since_activity
        self._send_prompt = send_prompt
        self._task: asyncio.Task[None] | None = None
        self._last_frame_size: int | None = None
        self._last_spoken_at: float = 0.0

    def start(self) -> None:
        """Start the background sampling loop if it isn't already running."""
        if self._task is not None and not self._task.done():
            return
        self._last_frame_size = None
        self._last_spoken_at = 0.0
        self._task = asyncio.create_task(self._run(), name="proactive-vision")

    async def stop(self) -> None:
        """Stop the background sampling loop."""
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
            await asyncio.sleep(self._config.sample_interval_seconds)
            try:
                await self._sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Proactive vision sampling failed: %s", e)

    async def _sample_once(self) -> None:
        jpeg_bytes = self._get_frame_jpeg()
        if jpeg_bytes is None:
            return

        frame_size = len(jpeg_bytes)
        previous_size = self._last_frame_size
        changed = (
            previous_size is not None
            and previous_size > 0
            and (abs(frame_size - previous_size) / previous_size >= _CHANGE_SIZE_RATIO)
        )
        self._last_frame_size = frame_size

        quiet_long_enough = self._seconds_since_activity() >= self._config.quiet_pause_seconds
        if not (changed or quiet_long_enough):
            return
        if not self._is_ready():
            return
        if time.monotonic() - self._last_spoken_at < self._config.speak_cooldown_seconds:
            return

        nudge = _NUDGE_ON_CHANGE if changed else _NUDGE_ON_QUIET
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        self._last_spoken_at = time.monotonic()
        await self._send_prompt(b64_image, nudge)
