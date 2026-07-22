import random
import asyncio
import logging
from typing import Any, Dict

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

# Body/head yaw used to "look away", matching sweep_look's convention.
_LOOK_AWAY_YAW = 0.9 * np.pi
_LOOK_AWAY_DURATION_S = 0.8
_GREEN_LIGHT_MIN_S = 1.0
_GREEN_LIGHT_MAX_S = 3.0
_TURN_BACK_DURATION_S = 0.4
_TURN_SETTLE_BUFFER_S = 0.1
_MOTION_SAMPLE_COUNT = 5
_MOTION_SAMPLE_INTERVAL_S = 0.12
_MOTION_DIFF_THRESHOLD = 8.0


class RedLightGreenLight(Tool):
    """Play one full round of Red Light, Green Light with the user."""

    name = "red_light_green_light"
    description = (
        "Play one round of Red Light, Green Light. Reachy turns away on its own (green light — the "
        "user is free to move), waits a short, randomized moment it decides itself, then whips back "
        "around (red light) and checks the camera for movement, returning caught=true if the user was "
        "still moving. Say something in character before calling it (e.g. 'Green light, go!') and "
        "narrate the result once you get it (call the user out if caught, praise them if still). Call "
        "this tool again for each new round, keep playing until the user asks to stop, then call "
        "move_head with direction='front' to face them again. Requires the camera; do not use if the "
        "camera is disabled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play one round: turn away, wait a random short time, turn back fast, and check for movement."""
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

        logger.info("Tool call: red_light_green_light")

        try:
            green_light_duration = random.uniform(_GREEN_LIGHT_MIN_S, _GREEN_LIGHT_MAX_S)
            await self._look_away(deps)
            await asyncio.sleep(green_light_duration)
            caught = await self._turn_and_check(deps)
            return {"status": "red light", "caught": caught, "green_light_seconds": round(green_light_duration, 1)}
        except Exception as e:
            logger.error("red_light_green_light failed")
            return {"error": f"red_light_green_light failed: {type(e).__name__}: {e}"}

    async def _look_away(self, deps: ToolDependencies) -> None:
        """Turn the head and body away so the user can move freely."""
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]

        look_away_pose = create_head_pose(0, 0, 0, 0, 0, _LOOK_AWAY_YAW, degrees=False)
        move = GotoQueueMove(
            target_head_pose=look_away_pose,
            start_head_pose=current_head_pose,
            target_antennas=(0, 0),
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=current_body_yaw + _LOOK_AWAY_YAW,
            start_body_yaw=current_body_yaw,
            duration=_LOOK_AWAY_DURATION_S,
        )
        deps.movement_manager.queue_move(move)
        deps.movement_manager.set_moving_state(_LOOK_AWAY_DURATION_S)
        await asyncio.sleep(_LOOK_AWAY_DURATION_S)

    async def _turn_and_check(self, deps: ToolDependencies) -> bool:
        """Turn back to face the user fast, then check the camera for movement. Returns True if caught."""
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]

        front_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
        move = GotoQueueMove(
            target_head_pose=front_pose,
            start_head_pose=current_head_pose,
            target_antennas=(0, 0),
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=0,
            start_body_yaw=current_body_yaw,
            duration=_TURN_BACK_DURATION_S,
        )
        deps.movement_manager.queue_move(move)
        deps.movement_manager.set_moving_state(_TURN_BACK_DURATION_S)

        await asyncio.sleep(_TURN_BACK_DURATION_S + _TURN_SETTLE_BUFFER_S)

        frames = []
        for i in range(_MOTION_SAMPLE_COUNT):
            frame = await asyncio.to_thread(deps.reachy_mini.media.get_frame)
            if frame is not None:
                frames.append(frame.astype(np.float32))
            if i < _MOTION_SAMPLE_COUNT - 1:
                await asyncio.sleep(_MOTION_SAMPLE_INTERVAL_S)

        if len(frames) < 2:
            raise RuntimeError("No frame available")

        max_diff = max(float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames)))
        return bool(max_diff > _MOTION_DIFF_THRESHOLD)
