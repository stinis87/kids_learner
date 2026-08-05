"""Tests for the Ring event watcher's config loading and reaction logic."""

import time
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.ring_client import RingEvent
from reachy_mini_conversation_app.ring_watcher import (
    RingWatcherConfig,
    RingWatcherEngine,
    load_config,
)


def test_load_config_enabled_by_default_without_marker_file(tmp_path: Path) -> None:
    """Profiles without a marker file still get the Ring watcher, using defaults."""
    assert load_config(tmp_path) == RingWatcherConfig()


def test_load_config_reads_custom_timings(tmp_path: Path) -> None:
    """Marker file values should override the defaults."""
    (tmp_path / "ring_watcher.txt").write_text("device_cooldown_seconds=30\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result == RingWatcherConfig(device_cooldown_seconds=30)


def test_load_config_falls_back_to_defaults_for_missing_keys(tmp_path: Path) -> None:
    """Unset keys should keep their default value."""
    (tmp_path / "ring_watcher.txt").write_text("door_call_mode=auto\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.door_call_mode == "auto"
    assert result.device_cooldown_seconds == RingWatcherConfig().device_cooldown_seconds


def test_load_config_returns_none_when_explicitly_disabled(tmp_path: Path) -> None:
    """A profile can opt out entirely with `enabled=false`."""
    (tmp_path / "ring_watcher.txt").write_text("enabled=false\n", encoding="utf-8")

    assert load_config(tmp_path) is None


def test_load_config_door_call_mode_defaults_to_ask(tmp_path: Path) -> None:
    """The doorbell asks by default rather than answering silently or automatically."""
    assert load_config(tmp_path) == RingWatcherConfig(door_call_mode="ask")


def test_load_config_reads_door_call_mode_auto(tmp_path: Path) -> None:
    """A profile can opt into fully autonomous answering."""
    (tmp_path / "ring_watcher.txt").write_text("door_call_mode=auto\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.door_call_mode == "auto"


def test_load_config_reads_door_call_mode_off(tmp_path: Path) -> None:
    """A profile can opt back into the original snapshot-only nudge, no call offered."""
    (tmp_path / "ring_watcher.txt").write_text("door_call_mode=off\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.door_call_mode == "off"


def test_load_config_ignores_unknown_door_call_mode(tmp_path: Path) -> None:
    """An unrecognized door_call_mode value is ignored, keeping the default."""
    (tmp_path / "ring_watcher.txt").write_text("door_call_mode=bogus\n", encoding="utf-8")

    result = load_config(tmp_path)

    assert result is not None
    assert result.door_call_mode == "ask"


def _event(device_name: str, event_id: int, kind: str = "motion") -> RingEvent:
    return RingEvent(device_name=device_name, event_id=event_id, kind=kind, created_at=None)  # type: ignore[arg-type]


def _fake_ring_client(snapshot: bytes = b"\xff\xd8jpeg\xff\xd9") -> MagicMock:
    client = MagicMock()
    client.async_start_event_listener = AsyncMock(return_value=True)
    client.async_stop_event_listener = AsyncMock()
    client.is_event_listener_active = MagicMock(return_value=True)
    client.async_get_device_snapshot = AsyncMock(return_value=snapshot)
    return client


@pytest.mark.asyncio
async def test_start_subscribes_to_the_push_listener_with_the_configured_client() -> None:
    """Starting the watcher subscribes to Ring push notifications via the client."""
    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
    )

    engine.start()
    await asyncio.sleep(0)  # let the listen task run once before stopping it
    await engine.stop()

    ring_client.async_start_event_listener.assert_awaited_once()
    ring_client.async_stop_event_listener.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_event_triggers_a_reaction() -> None:
    """A pushed event should fetch a snapshot and prompt the model to react."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )

    await engine._react_to_event(_event("Garden", 1))

    assert len(sent) == 1
    assert "Garden" in sent[0][1]
    ring_client.async_get_device_snapshot.assert_awaited_once_with("Garden")


@pytest.mark.asyncio
async def test_new_event_triggers_play_emotion_reaction() -> None:
    """A new motion event should also queue a physical emotion reaction, not just speech."""
    played: list[str] = []

    async def play_emotion(emotion: str) -> None:
        played.append(emotion)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
        play_emotion=play_emotion,
    )

    await engine._react_to_event(_event("Garden", 1, kind="motion"))

    assert played == ["attentive"]


@pytest.mark.asyncio
async def test_ding_event_triggers_excited_emotion() -> None:
    """A doorbell ding should play a more excited reaction than plain motion."""
    played: list[str] = []

    async def play_emotion(emotion: str) -> None:
        played.append(emotion)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
        play_emotion=play_emotion,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))

    assert played == ["excited"]


@pytest.mark.asyncio
async def test_play_emotion_failure_does_not_block_the_spoken_reaction() -> None:
    """A broken emotion callback shouldn't stop the image nudge prompt from still going out."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def failing_play_emotion(emotion: str) -> None:
        raise RuntimeError("movement manager unavailable")

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        play_emotion=failing_play_emotion,
    )

    await engine._react_to_event(_event("Garden", 1))

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_ding_event_uses_doorbell_nudge_wording() -> None:
    """A 'ding' event should produce doorbell-specific wording, not the generic motion one."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0, door_call_mode="off"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))

    assert "doorbell" in sent[0][1]


@pytest.mark.asyncio
async def test_device_cooldown_suppresses_rapid_repeat_reactions() -> None:
    """A device that just reacted shouldn't react again within its cooldown window."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=120),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    engine._last_reacted_at["Garden"] = time.monotonic()

    await engine._react_to_event(_event("Garden", 2))

    assert sent == []


@pytest.mark.asyncio
async def test_ding_bypasses_cooldown_from_a_recent_motion_reaction() -> None:
    """A doorbell ring shouldn't be suppressed by a cooldown started by a prior motion event."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=120, door_call_mode="off"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
    )
    engine._last_reacted_at["Front Door"] = time.monotonic()  # simulate the just-reacted motion event

    await engine._react_to_event(_event("Front Door", 2, kind="ding"))

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_not_ready_suppresses_reaction() -> None:
    """A pending model response should hold off the Ring watcher's prompt."""
    sent: list[tuple[str, str]] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: False,
        send_prompt=send_prompt,
    )

    await engine._react_to_event(_event("Garden", 1))

    assert sent == []


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_crash_the_reaction() -> None:
    """A failed snapshot fetch shouldn't raise out of the reaction handler."""
    ring_client = _fake_ring_client()
    ring_client.async_get_device_snapshot = AsyncMock(side_effect=RuntimeError("device asleep"))
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
    )

    await engine._react_to_event(_event("Garden", 1))  # must not raise


@pytest.mark.asyncio
async def test_ding_event_answers_the_call_when_door_call_mode_is_auto() -> None:
    """A 'ding' with door_call_mode=auto should open a call instead of sending an image nudge."""
    sent: list[tuple[str, str]] = []
    answered: list[str] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def answer_ding(device_name: str) -> None:
        answered.append(device_name)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0, door_call_mode="auto"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        answer_ding=answer_ding,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))

    assert answered == ["Front Door"]
    assert sent == []
    ring_client.async_get_device_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_ding_event_asks_before_answering_by_default() -> None:
    """The default door_call_mode='ask' nudges the model to ask before opening a call itself."""
    sent: list[tuple[str, str]] = []
    answered: list[str] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def answer_ding(device_name: str) -> None:
        answered.append(device_name)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        answer_ding=answer_ding,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))

    assert answered == []
    assert len(sent) == 1
    assert "talk_to_door" in sent[0][1]


@pytest.mark.asyncio
async def test_ding_event_falls_back_to_plain_image_nudge_when_door_call_mode_is_off() -> None:
    """With door_call_mode='off', a 'ding' just gets the original image nudge, no question asked."""
    sent: list[tuple[str, str]] = []
    answered: list[str] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def answer_ding(device_name: str) -> None:
        answered.append(device_name)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0, door_call_mode="off"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        answer_ding=answer_ding,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))

    assert answered == []
    assert len(sent) == 1
    assert "talk_to_door" not in sent[0][1]


@pytest.mark.asyncio
async def test_motion_event_never_answers_the_call_even_when_door_call_mode_is_auto() -> None:
    """door_call_mode only applies to 'ding' events; plain motion still just gets a nudge."""
    sent: list[tuple[str, str]] = []
    answered: list[str] = []

    async def send_prompt(b64_image: str, nudge: str) -> None:
        sent.append((b64_image, nudge))

    async def answer_ding(device_name: str) -> None:
        answered.append(device_name)

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0, door_call_mode="auto"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=send_prompt,
        answer_ding=answer_ding,
    )

    await engine._react_to_event(_event("Garden", 1, kind="motion"))

    assert answered == []
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_answer_ding_failure_does_not_crash_the_poll_loop() -> None:
    """A broken auto-answer callback is logged and swallowed, not raised."""

    async def answer_ding(device_name: str) -> None:
        raise RuntimeError("Ring rejected the call")

    ring_client = _fake_ring_client()
    engine = RingWatcherEngine(
        RingWatcherConfig(device_cooldown_seconds=0, door_call_mode="auto"),
        ring_client=ring_client,
        is_ready=lambda: True,
        send_prompt=AsyncMock(),
        answer_ding=answer_ding,
    )

    await engine._react_to_event(_event("Front Door", 1, kind="ding"))  # must not raise
