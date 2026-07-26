import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_speaker_tracking_ignored_when_head_tracking_is_off() -> None:
    """DoA polling should be skipped entirely while head tracking isn't active."""
    robot = MagicMock()
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = False

    manager._manage_speaker_tracking(manager._now())

    robot.media.get_DoA.assert_not_called()
    robot.get_tracked_face.assert_not_called()


def test_speaker_tracking_defers_once_a_face_is_locked() -> None:
    """A locked face with speech still coming from roughly the same direction stays put."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = True
    robot.media.get_DoA.return_value = (np.pi / 2, True)  # angle=front, matches current facing
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._last_doa_poll_time = 0.0
    manager._speaker_yaw_current = 0.0

    manager._manage_speaker_tracking(manager._now())

    robot.get_tracked_face.assert_called_once_with(wait=False)
    assert manager._speaker_yaw_target == pytest.approx(0.0)
    assert manager._pending_speaker_yaw_target is None


def test_speaker_tracking_ignores_brief_off_axis_speech_while_face_locked() -> None:
    """A single off-axis DoA reading shouldn't immediately redirect away from a locked face."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = True
    robot.media.get_DoA.return_value = (np.pi, True)  # angle=right, far from current front-facing
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._last_doa_poll_time = 0.0
    manager._speaker_yaw_current = 0.0

    manager._manage_speaker_tracking(manager._now())

    # Not yet confirmed: yaw target should stay on the locked face, only pending is armed.
    assert manager._speaker_yaw_target == pytest.approx(0.0)
    assert manager._pending_speaker_yaw_target is not None


def test_speaker_tracking_redirects_to_new_speaker_after_confirm_window() -> None:
    """A persistently different speech direction should override a stale face lock."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = True
    robot.media.get_DoA.return_value = (np.pi, True)  # angle=right, far from current front-facing
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._speaker_yaw_current = 0.0

    now = manager._now()
    manager._last_doa_poll_time = now - manager._doa_poll_interval_s
    manager._manage_speaker_tracking(now)
    assert manager._speaker_yaw_target == pytest.approx(0.0)

    later = now + manager._speaker_redirect_confirm_s + manager._doa_poll_interval_s
    manager._last_doa_poll_time = later - manager._doa_poll_interval_s
    manager._manage_speaker_tracking(later)

    assert manager._speaker_yaw_target == pytest.approx(-np.pi / 2)
    assert manager._pending_speaker_yaw_target is None


def test_speaker_tracking_sets_yaw_target_toward_active_speaker() -> None:
    """A DoA reading with detected speech and no locked face should set a yaw target."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = False
    robot.media.get_DoA.return_value = (0.0, True)  # angle=0 (left), speech detected
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._last_doa_poll_time = 0.0

    manager._manage_speaker_tracking(manager._now())

    assert manager._speaker_yaw_target == pytest.approx(np.pi / 2)


def test_speaker_tracking_ignores_doa_without_detected_speech() -> None:
    """A DoA reading without detected speech shouldn't move the yaw target."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = False
    robot.media.get_DoA.return_value = (0.0, False)
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._last_doa_poll_time = 0.0

    manager._manage_speaker_tracking(manager._now())

    assert manager._speaker_yaw_target == 0.0


def test_speaker_tracking_noop_without_respeaker_hardware() -> None:
    """A missing mic array (get_DoA returns None) should be a graceful no-op."""
    robot = MagicMock()
    robot.get_tracked_face.return_value.detected = False
    robot.media.get_DoA.return_value = None
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._head_tracking = True
    manager._last_doa_poll_time = 0.0

    manager._manage_speaker_tracking(manager._now())

    assert manager._speaker_yaw_target == 0.0


def test_blend_speaker_yaw_disabled_returns_body_yaw_unchanged() -> None:
    """With speaker tracking disabled, body yaw should pass through untouched."""
    robot = MagicMock()
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = False

    assert manager._blend_speaker_yaw(manager._now(), 0.3) == 0.3


def test_blend_speaker_yaw_steers_toward_target_over_time() -> None:
    """Enabled speaker tracking should gradually steer body yaw toward the DoA target."""
    robot = MagicMock()
    manager = MovementManager(robot)
    manager._speaker_tracking_enabled = True
    manager._speaker_yaw_target = 1.0
    manager._speaker_yaw_current = 0.0
    manager._last_speaker_yaw_blend_time = manager._now() - manager._speaker_yaw_blend_duration_s

    body_yaw = manager._blend_speaker_yaw(manager._now(), 0.0)

    assert body_yaw == pytest.approx(1.0, abs=1e-6)


def test_disabling_speaker_tracking_resets_yaw_state() -> None:
    """Turning speaker tracking off should reset any accumulated yaw bias."""
    robot = MagicMock()
    manager = MovementManager(robot)
    manager._speaker_yaw_target = 1.0
    manager._speaker_yaw_current = 0.5
    manager._pending_speaker_yaw_target = 2.0

    manager._handle_command("set_speaker_tracking", False, manager._now())

    assert manager._speaker_tracking_enabled is False
    assert manager._speaker_yaw_target == 0.0
    assert manager._speaker_yaw_current == 0.0
    assert manager._pending_speaker_yaw_target is None
