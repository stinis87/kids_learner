import math
import random
import asyncio
import logging
from typing import Any, Dict, Tuple

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

# Timings and thresholds mirror the "expert mode" of this game
# (ChristmasTree/expert-red-light-green-light): quick head-hide/reveal snaps, antennas
# that act alert or coy, occasional fake-out green lights, and a long, randomized watch
# window rather than a fixed one, so the pacing itself keeps players honest. On top of
# that we add a full body turn: Reachy spins its back to the user on green light and
# spins back to face them the instant it's red light, like the real playground game.
_HIDE_DURATION_S = 0.5  # time to spin away and tuck the head down
_REVEAL_DURATION_S = 0.35  # time to spin back and snap the head up, quicker than hiding
_RED_SETTLE_S = 0.3  # pause after snapping up before scanning, to let motion blur clear
_GREEN_LIGHT_MIN_S = 0.8
_GREEN_LIGHT_MAX_S = 2.4
_FAKEOUT_PROB = 0.08
_FAKEOUT_GREEN_S = 0.45
_RED_LIGHT_MIN_S = 3.0
_RED_LIGHT_MAX_S = 7.0
_MOTION_SAMPLE_INTERVAL_S = 0.12
# Mean per-pixel brightness change between consecutive frames; above this, someone moved.
_MOTION_DIFF_THRESHOLD = 8.0
_CAUGHT_TURN_DURATION_S = 0.3
_CAUGHT_YAW_RANGE_DEG = (-80.0, 80.0)
# How far Reachy spins its body away from the user on green light, in degrees. Kept just
# under 180 to stay clear of the body yaw's hard end-stop.
_BODY_TURN_AWAY_DEG = 175.0

# Antenna poses double as Reachy's facial expression here: relaxed while hiding its
# eyes, perked up and alert while watching, cocked sideways when calling someone out.
_ANTENNAS_HIDDEN: Tuple[float, float] = (0.0, 0.0)
_ANTENNAS_WATCHING: Tuple[float, float] = (0.6, 0.6)
_ANTENNAS_CALLING_OUT: Tuple[float, float] = (0.5, -0.5)


class RedLightGreenLight(Tool):
    """Play one full round of Red Light, Green Light with the user."""

    name = "red_light_green_light"
    description = (
        "Play one full round of Red Light, Green Light per call — this single call covers both the "
        "green light and red light phases, so never split a round across two calls and never wait for "
        "the user to say anything in between. Say 'Green light!' out loud AS you call this tool. Reachy "
        "then spins its body 180 degrees to turn its back on the user and tucks its head down to hide "
        "its eyes (they may move now) for a short randomized moment, occasionally a lightning-fast "
        "fake-out, before quickly spinning back around on its own to face the user, snapping its head "
        "back up, and watching the camera for several seconds — turning to look right at anyone it "
        "catches moving. The call does not return until all of that has finished, so as soon as it "
        "returns say 'Red light!' plus the verdict right away (call them out if caught=true, praise them "
        "if still). Call this tool again immediately to start the next round for as many rounds as the "
        "user wants. When the user asks to stop, call move_head with direction='front' to face them "
        "again. Requires the camera; do not use if the camera is disabled. Works no matter what language "
        "the user asks in — e.g. Norwegian 'kan vi spille rødt lys, grønt lys' or 'stopp og gå' request "
        "the same game as their English equivalent."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run one full green-light/red-light round."""
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

        logger.info("Tool call: red_light_green_light")

        try:
            green_light_result = await self._hide_and_wait(deps)
            red_light_result = await self._watch_for_movement(deps)
            return {**green_light_result, **red_light_result}
        except Exception as e:
            logger.error("red_light_green_light failed")
            return {"error": f"red_light_green_light failed: {type(e).__name__}: {e}"}

    async def _hide_and_wait(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Spin the body away from the user and tuck the head down, then wait a randomized moment."""
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]

        move = GotoQueueMove(
            target_head_pose=SLEEP_HEAD_POSE.astype(np.float32),
            start_head_pose=current_head_pose,
            target_antennas=_ANTENNAS_HIDDEN,
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=math.radians(_BODY_TURN_AWAY_DEG),
            start_body_yaw=current_body_yaw,
            duration=_HIDE_DURATION_S,
        )
        deps.movement_manager.queue_move(move)
        deps.movement_manager.set_moving_state(_HIDE_DURATION_S)
        await asyncio.sleep(_HIDE_DURATION_S)

        is_fakeout = random.random() < _FAKEOUT_PROB
        green_light_duration = (
            _FAKEOUT_GREEN_S if is_fakeout else random.uniform(_GREEN_LIGHT_MIN_S, _GREEN_LIGHT_MAX_S)
        )
        await asyncio.sleep(green_light_duration)

        return {
            "green_light_seconds": round(green_light_duration, 2),
            "fakeout": is_fakeout,
        }

    async def _watch_for_movement(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Spin back to face the user, snap the head up, watch for movement, and turn to look at whoever gets caught."""
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]

        front_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
        move = GotoQueueMove(
            target_head_pose=front_pose,
            start_head_pose=current_head_pose,
            target_antennas=_ANTENNAS_WATCHING,
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=0,
            start_body_yaw=current_body_yaw,
            duration=_REVEAL_DURATION_S,
        )
        deps.movement_manager.queue_move(move)
        deps.movement_manager.set_moving_state(_REVEAL_DURATION_S)
        await asyncio.sleep(_REVEAL_DURATION_S + _RED_SETTLE_S)

        caught_x, frame_width, frames_read = await self._scan_for_movement(deps)

        if frames_read < 2:
            raise RuntimeError("No frame available")

        caught = caught_x is not None
        if caught_x is not None and frame_width is not None:
            await self._turn_to_caught_player(deps, caught_x, frame_width)

        return {"status": "red light", "caught": caught}

    async def _scan_for_movement(self, deps: ToolDependencies) -> tuple[float | None, int | None, int]:
        """Watch the camera and return the pixel x-position of the first movement caught, if any.

        Diffs each frame against the previous one and flags movement once the mean per-pixel
        brightness change crosses a threshold, weighting where in the frame that change is
        concentrated to know which way to turn. Stops as soon as someone is caught instead of
        always running the full scan window, matching how the expert-mode game snaps to a mover
        the instant they slip.
        """
        scan_for = random.uniform(_RED_LIGHT_MIN_S, _RED_LIGHT_MAX_S)
        previous_frame: np.ndarray | None = None
        frames_read = 0
        frame_width: int | None = None
        elapsed = 0.0

        while elapsed < scan_for:
            frame = await asyncio.to_thread(deps.reachy_mini.media.get_frame)
            if frame is not None:
                frames_read += 1
                frame_width = frame.shape[1]
                if previous_frame is not None and previous_frame.shape == frame.shape:
                    diff = np.abs(frame.astype(np.float32) - previous_frame.astype(np.float32)).mean(axis=-1)
                    if diff.mean() > _MOTION_DIFF_THRESHOLD:
                        column_weight = diff.sum(axis=0)
                        centroid_x = float(np.average(np.arange(frame_width), weights=column_weight))
                        return centroid_x, frame_width, frames_read
                previous_frame = frame

            await asyncio.sleep(_MOTION_SAMPLE_INTERVAL_S)
            elapsed += _MOTION_SAMPLE_INTERVAL_S

        return None, frame_width, frames_read

    async def _turn_to_caught_player(self, deps: ToolDependencies, caught_x: float, frame_width: int) -> None:
        """Turn the head to look at the pixel position of whoever just got caught moving."""
        normalized_x = ((caught_x / frame_width) - 0.5) * 2
        lo, hi = _CAUGHT_YAW_RANGE_DEG
        target_yaw_deg = normalized_x * (hi - lo) / 2 + (hi + lo) / 2

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()

        turn_pose = create_head_pose(yaw=-target_yaw_deg, degrees=True)
        move = GotoQueueMove(
            target_head_pose=turn_pose,
            start_head_pose=current_head_pose,
            target_antennas=_ANTENNAS_CALLING_OUT,
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=0,
            start_body_yaw=head_joints[0],
            duration=_CAUGHT_TURN_DURATION_S,
        )
        deps.movement_manager.queue_move(move)
        deps.movement_manager.set_moving_state(_CAUGHT_TURN_DURATION_S)
        await asyncio.sleep(_CAUGHT_TURN_DURATION_S)
