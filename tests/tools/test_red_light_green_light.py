"""Tests for the red_light_green_light tool."""

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


@pytest.mark.asyncio
async def test_round_queues_look_away_then_turn_back_moves() -> None:
    """A round queues the look-away move first and the turn-back move second."""
    deps = _make_deps()

    with patch("asyncio.sleep", new=AsyncMock()):
        await RedLightGreenLight()(deps)

    assert deps.movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
async def test_round_reports_caught_when_frames_differ() -> None:
    """A round flags caught=True when consecutive frames differ a lot."""
    deps = _make_deps()
    frame_a = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 255, dtype=np.uint8)
    deps.reachy_mini.media.get_frame.side_effect = [frame_a, frame_b, frame_a, frame_b, frame_a]

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await RedLightGreenLight()(deps)

    assert result["caught"] is True


@pytest.mark.asyncio
async def test_round_reports_not_caught_when_frames_are_identical() -> None:
    """A round flags caught=False when consecutive frames barely change."""
    deps = _make_deps()
    still_frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    deps.reachy_mini.media.get_frame.return_value = still_frame

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await RedLightGreenLight()(deps)

    assert result["caught"] is False


@pytest.mark.asyncio
async def test_round_uses_a_randomized_green_light_duration() -> None:
    """Each round waits for a different, randomized green-light duration."""
    deps = _make_deps()

    with patch("random.uniform", return_value=2.5) as mock_uniform, patch("asyncio.sleep", new=AsyncMock()):
        result = await RedLightGreenLight()(deps)

    mock_uniform.assert_called_once()
    assert result["green_light_seconds"] == 2.5


@pytest.mark.asyncio
async def test_reports_error_when_camera_disabled() -> None:
    """The tool reports an error and never reads a frame when the camera is disabled."""
    deps = _make_deps(camera_enabled=False)

    result = await RedLightGreenLight()(deps)

    assert "error" in result
    deps.reachy_mini.media.get_frame.assert_not_called()


@pytest.mark.asyncio
async def test_reports_error_when_no_frame_available() -> None:
    """The tool reports an error when the camera never returns a frame."""
    deps = _make_deps()
    deps.reachy_mini.media.get_frame.return_value = None

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await RedLightGreenLight()(deps)

    assert "error" in result
