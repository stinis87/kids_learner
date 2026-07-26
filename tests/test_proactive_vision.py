import time
from pathlib import Path

import pytest

from reachy_mini_conversation_app.proactive_vision import (
    ProactiveVisionConfig,
    ProactiveVisionEngine,
    load_config,
)


def test_load_config_returns_none_when_profile_does_not_opt_in(tmp_path: Path) -> None:
    """Profiles without a marker file should not enable proactive vision."""
    assert load_config(tmp_path) is None


def test_load_config_reads_custom_timings(tmp_path: Path) -> None:
    """Marker file values should override the defaults."""
    (tmp_path / "proactive_vision.txt").write_text(
        "sample_interval_seconds=2\nquiet_pause_seconds=5\nspeak_cooldown_seconds=8\n",
        encoding="utf-8",
    )

    result = load_config(tmp_path)

    assert result == ProactiveVisionConfig(
        sample_interval_seconds=2,
        quiet_pause_seconds=5,
        speak_cooldown_seconds=8,
    )


def test_load_config_falls_back_to_defaults_for_missing_keys(tmp_path: Path) -> None:
    """Unset keys should keep their default value."""
    (tmp_path / "proactive_vision.txt").write_text("quiet_pause_seconds=9\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.quiet_pause_seconds == 9
    assert result.sample_interval_seconds == ProactiveVisionConfig().sample_interval_seconds


@pytest.mark.asyncio
async def test_sample_once_speaks_up_on_meaningful_size_change() -> None:
    """A large jump in JPEG size (proxy for a page turn) should trigger a prompt."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    engine = ProactiveVisionEngine(
        ProactiveVisionConfig(quiet_pause_seconds=999, speak_cooldown_seconds=0),
        get_frame_jpeg=lambda: b"x" * 1000,
        is_ready=lambda: True,
        seconds_since_activity=lambda: 0.0,
        send_prompt=send_prompt,
    )

    await engine._sample_once()  # establishes the baseline frame size, no trigger yet
    assert sent == []

    engine._get_frame_jpeg = lambda: b"x" * 5000  # large jump => "page changed"
    await engine._sample_once()

    assert len(sent) == 1
    assert "changed" in sent[0][1]


@pytest.mark.asyncio
async def test_sample_once_speaks_up_after_quiet_pause() -> None:
    """A long quiet period should trigger a prompt even without a visual change."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    engine = ProactiveVisionEngine(
        ProactiveVisionConfig(quiet_pause_seconds=5, speak_cooldown_seconds=0),
        get_frame_jpeg=lambda: b"same-frame",
        is_ready=lambda: True,
        seconds_since_activity=lambda: 10.0,
        send_prompt=send_prompt,
    )

    await engine._sample_once()

    assert len(sent) == 1
    assert "quiet" in sent[0][1]


@pytest.mark.asyncio
async def test_sample_once_respects_cooldown_after_speaking() -> None:
    """Reachy shouldn't speak again immediately after just having spoken."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    engine = ProactiveVisionEngine(
        ProactiveVisionConfig(quiet_pause_seconds=0, speak_cooldown_seconds=60),
        get_frame_jpeg=lambda: b"frame",
        is_ready=lambda: True,
        seconds_since_activity=lambda: 100.0,
        send_prompt=send_prompt,
    )
    engine._last_spoken_at = time.monotonic()

    await engine._sample_once()

    assert sent == []


@pytest.mark.asyncio
async def test_sample_once_waits_for_ready_state() -> None:
    """A pending model response should hold off the proactive prompt."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    engine = ProactiveVisionEngine(
        ProactiveVisionConfig(quiet_pause_seconds=0, speak_cooldown_seconds=0),
        get_frame_jpeg=lambda: b"frame",
        is_ready=lambda: False,
        seconds_since_activity=lambda: 100.0,
        send_prompt=send_prompt,
    )

    await engine._sample_once()

    assert sent == []
