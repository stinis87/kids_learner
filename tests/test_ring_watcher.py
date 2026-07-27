"""Tests for the Ring event watcher's config loading and reaction logic."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.ring_client import RingEvent
from reachy_mini_conversation_app.ring_watcher import (
    RingWatcherConfig,
    RingWatcherEngine,
    load_config,
)


def test_load_config_returns_none_when_profile_does_not_opt_in(tmp_path: Path) -> None:
    """Profiles without a marker file should not enable the Ring watcher."""
    assert load_config(tmp_path) is None


def test_load_config_reads_custom_timings(tmp_path: Path) -> None:
    """Marker file values should override the defaults."""
    (tmp_path / "ring_watcher.txt").write_text(
        "poll_interval_seconds=5\ndevice_cooldown_seconds=30\n",
        encoding="utf-8",
    )

    result = load_config(tmp_path)

    assert result == RingWatcherConfig(poll_interval_seconds=5, device_cooldown_seconds=30)


def test_load_config_falls_back_to_defaults_for_missing_keys(tmp_path: Path) -> None:
    """Unset keys should keep their default value."""
    (tmp_path / "ring_watcher.txt").write_text("poll_interval_seconds=15\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.poll_interval_seconds == 15
    assert result.device_cooldown_seconds == RingWatcherConfig().device_cooldown_seconds


def _event(device_name: str, event_id: int, kind: str = "motion") -> RingEvent:
    return RingEvent(device_name=device_name, event_id=event_id, kind=kind, created_at=None)  # type: ignore[arg-type]


def _fake_ring_client(events: dict[str, RingEvent], snapshot: bytes = b"\xff\xd8jpeg\xff\xd9") -> MagicMock:
    client = MagicMock()
    client.async_get_latest_events = AsyncMock(return_value=events)
    client.async_get_device_snapshot = AsyncMock(return_value=snapshot)
    return client


@pytest.mark.asyncio
async def test_first_poll_seeds_without_reacting() -> None:
    """The very first poll should record existing events but never react to them."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )

    await engine._poll_once()

    assert sent == []
    ring_client.async_get_device_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_event_after_seeding_triggers_a_reaction() -> None:
    """A genuinely new event id after the seeding poll should fetch a snapshot and react."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2)})
    await engine._poll_once()

    assert len(sent) == 1
    assert "Garden" in sent[0][1]
    ring_client.async_get_device_snapshot.assert_awaited_once_with("Garden")


@pytest.mark.asyncio
async def test_new_event_triggers_play_emotion_reaction() -> None:
    """A new motion event should also queue a physical emotion reaction, not just speech."""
    played: list[str] = []

    async def play_emotion(emotion: str) -> None:
        played.append(emotion)

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
        play_emotion=play_emotion,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2, kind="motion")})
    await engine._poll_once()

    assert played == ["attentive"]


@pytest.mark.asyncio
async def test_ding_event_triggers_excited_emotion() -> None:
    """A doorbell ding should play a more excited reaction than plain motion."""
    played: list[str] = []

    async def play_emotion(emotion: str) -> None:
        played.append(emotion)

    ring_client = _fake_ring_client({"Front Door": _event("Front Door", 1, kind="ding")})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
        play_emotion=play_emotion,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Front Door": _event("Front Door", 2, kind="ding")})
    await engine._poll_once()

    assert played == ["excited"]


@pytest.mark.asyncio
async def test_play_emotion_failure_does_not_block_the_spoken_reaction() -> None:
    """A broken emotion callback shouldn't stop the image nudge prompt from still going out."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def failing_play_emotion(emotion: str) -> None:
        raise RuntimeError("movement manager unavailable")

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        play_emotion=failing_play_emotion,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2)})
    await engine._poll_once()

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_ding_event_uses_doorbell_nudge_wording() -> None:
    """A 'ding' event should produce doorbell-specific wording, not the generic motion one."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Front Door": _event("Front Door", 1, kind="ding")})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Front Door": _event("Front Door", 2, kind="ding")})
    await engine._poll_once()

    assert "doorbell" in sent[0][1]


@pytest.mark.asyncio
async def test_same_event_id_does_not_react_again() -> None:
    """Polling again with an unchanged event id must not re-trigger a reaction."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll
    await engine._poll_once()  # same event id again

    assert sent == []


@pytest.mark.asyncio
async def test_device_cooldown_suppresses_rapid_repeat_reactions() -> None:
    """A device that just reacted shouldn't react again within its cooldown window."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=120),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll
    engine._last_reacted_at["Garden"] = time.monotonic()

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2)})
    await engine._poll_once()

    assert sent == []


@pytest.mark.asyncio
async def test_not_ready_suppresses_reaction() -> None:
    """A pending model response should hold off the Ring watcher's prompt."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: False,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2)})
    await engine._poll_once()

    assert sent == []


@pytest.mark.asyncio
async def test_snapshot_failure_for_one_device_does_not_block_reaction_tracking() -> None:
    """A failed snapshot fetch shouldn't crash the poll loop or corrupt event tracking."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client({"Garden": _event("Garden", 1)})
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    await engine._poll_once()  # seeding poll

    ring_client.async_get_latest_events = AsyncMock(return_value={"Garden": _event("Garden", 2)})
    ring_client.async_get_device_snapshot = AsyncMock(side_effect=RuntimeError("device asleep"))
    await engine._poll_once()

    assert sent == []
