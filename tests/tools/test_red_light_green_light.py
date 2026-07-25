"""Tests for the red_light_green_light tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.red_light_green_light import RedLightGreenLight


def _make_deps(camera_enabled: bool = True) -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)

    return ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=camera_enabled,
    )


def _patch_detector(landmarks_per_frame: list[list[list[tuple[int, int]]]]) -> Any:
    """Patch tool detector construction so it returns one people-list per sampled frame."""
    detector = MagicMock()
    detector.detect.side_effect = landmarks_per_frame
    return patch.object(RedLightGreenLight, "_get_detector", new=AsyncMock(return_value=detector))


@pytest.mark.asyncio
async def test_green_light_queues_a_move_and_waits_a_randomized_duration() -> None:
    """The green_light phase queues a hide move and waits its own randomized moment."""
    deps = _make_deps()

    with (
        patch("random.random", return_value=1.0),  # above the fake-out probability
        patch("random.uniform", return_value=2.5) as mock_uniform,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await RedLightGreenLight()(deps, phase="green_light")

    deps.movement_manager.queue_move.assert_called_once()
    mock_uniform.assert_called_once()
    assert result["green_light_seconds"] == 2.5
    assert result["fakeout"] is False


@pytest.mark.asyncio
async def test_green_light_can_trigger_a_fakeout() -> None:
    """A lucky roll below the fake-out probability produces a lightning-fast green light."""
    deps = _make_deps()

    with (
        patch("random.random", return_value=0.0),  # below the fake-out probability
        patch("random.uniform") as mock_uniform,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await RedLightGreenLight()(deps, phase="green_light")

    mock_uniform.assert_not_called()
    assert result["fakeout"] is True
    assert result["green_light_seconds"] == 0.45


@pytest.mark.asyncio
async def test_red_light_reports_caught_and_turns_to_the_mover() -> None:
    """The red_light phase flags caught=True and turns toward whoever's landmarks shifted a lot."""
    deps = _make_deps()
    still = [(10, 10), (20, 20)]
    moved = [(200, 200), (220, 220)]
    frames = [[still], [still], [still], [still], [moved]]

    with (
        patch("random.uniform", return_value=0.6),  # short scan window matching the 5 sampled frames
        patch("asyncio.sleep", new=AsyncMock()),
        _patch_detector(frames),
    ):
        result = await RedLightGreenLight()(deps, phase="red_light")

    assert result["caught"] is True
    # One move to snap the head up, one move to turn and look at the caught player.
    assert deps.movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
async def test_red_light_reports_not_caught_when_nobody_moves() -> None:
    """The red_light phase flags caught=False when a detected person barely moves."""
    deps = _make_deps()
    still = [(10, 10), (20, 20)]
    frames = [[still]] * 5

    with (
        patch("random.uniform", return_value=0.6),
        patch("asyncio.sleep", new=AsyncMock()),
        _patch_detector(frames),
    ):
        result = await RedLightGreenLight()(deps, phase="red_light")

    assert result["caught"] is False
    # Only the snap-up move; nobody to turn toward.
    assert deps.movement_manager.queue_move.call_count == 1


@pytest.mark.asyncio
async def test_reports_error_when_camera_disabled() -> None:
    """Both phases report an error and never read a frame when the camera is disabled."""
    deps = _make_deps(camera_enabled=False)

    result = await RedLightGreenLight()(deps, phase="red_light")

    assert "error" in result
    deps.reachy_mini.media.get_frame.assert_not_called()


@pytest.mark.asyncio
async def test_red_light_reports_error_when_no_frame_available() -> None:
    """The red_light phase reports an error when the camera never returns a frame."""
    deps = _make_deps()
    deps.reachy_mini.media.get_frame.return_value = None

    with (
        patch("random.uniform", return_value=0.3),
        patch("asyncio.sleep", new=AsyncMock()),
        _patch_detector([]),
    ):
        result = await RedLightGreenLight()(deps, phase="red_light")

    assert "error" in result


@pytest.mark.asyncio
async def test_invalid_phase_reports_error() -> None:
    """An unknown phase value reports an error instead of running a phase."""
    deps = _make_deps()

    result = await RedLightGreenLight()(deps, phase="blue_light")

    assert "error" in result
