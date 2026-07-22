"""Tests for the red_light_green_light tool."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.red_light_green_light import RedLightGreenLight


def _make_deps(camera_enabled: bool = True) -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])

    return ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=camera_enabled,
    )


@pytest.mark.asyncio
async def test_green_light_queues_a_move_and_returns_without_error() -> None:
    """The green_light phase queues a look-away move and reports success."""
    deps = _make_deps()

    result = await RedLightGreenLight()(deps, phase="green_light")

    assert "error" not in result
    deps.movement_manager.queue_move.assert_called_once()
    deps.movement_manager.set_moving_state.assert_called_once()


@pytest.mark.asyncio
async def test_red_light_reports_caught_when_frames_differ() -> None:
    """The red_light phase flags caught=True when consecutive frames differ a lot."""
    deps = _make_deps()
    frame_a = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 255, dtype=np.uint8)
    deps.reachy_mini.media.get_frame.side_effect = [frame_a, frame_b, frame_a, frame_b, frame_a]

    result = await RedLightGreenLight()(deps, phase="red_light")

    assert result["caught"] is True


@pytest.mark.asyncio
async def test_red_light_reports_not_caught_when_frames_are_identical() -> None:
    """The red_light phase flags caught=False when consecutive frames barely change."""
    deps = _make_deps()
    still_frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    deps.reachy_mini.media.get_frame.return_value = still_frame

    result = await RedLightGreenLight()(deps, phase="red_light")

    assert result["caught"] is False


@pytest.mark.asyncio
async def test_red_light_reports_error_when_camera_disabled() -> None:
    """The red_light phase reports an error and never reads a frame when the camera is disabled."""
    deps = _make_deps(camera_enabled=False)

    result = await RedLightGreenLight()(deps, phase="red_light")

    assert "error" in result
    deps.reachy_mini.media.get_frame.assert_not_called()


@pytest.mark.asyncio
async def test_red_light_reports_error_when_no_frame_available() -> None:
    """The red_light phase reports an error when the camera never returns a frame."""
    deps = _make_deps()
    deps.reachy_mini.media.get_frame.return_value = None

    result = await RedLightGreenLight()(deps, phase="red_light")

    assert "error" in result


@pytest.mark.asyncio
async def test_invalid_phase_reports_error() -> None:
    """An unknown phase value reports an error instead of running a phase."""
    deps = _make_deps()

    result = await RedLightGreenLight()(deps, phase="blue_light")

    assert "error" in result
