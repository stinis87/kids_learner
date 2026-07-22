import asyncio
import logging
from typing import Any, Dict, Literal

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Phase = Literal["green_light", "red_light"]

# Body/head yaw used to "look away", matching sweep_look's convention.
_LOOK_AWAY_YAW = 0.9 * np.pi
_LOOK_AWAY_DURATION_S = 1.2
_TURN_BACK_DURATION_S = 0.4
_TURN_SETTLE_BUFFER_S = 0.1
_MOTION_SAMPLE_COUNT = 5
_MOTION_SAMPLE_INTERVAL_S = 0.12
_MOTION_DIFF_THRESHOLD = 8.0


class RedLightGreenLight(Tool):
    """Play one round-phase of Red Light, Green Light with the user."""

    name = "red_light_green_light"
    description = (
        "Play one phase of Red Light, Green Light. Call with phase='green_light' after you say "
        "'Green light!' out loud: Reachy turns away so the user is free to move. Later, say "
        "'Red light!' out loud and call with phase='red_light': Reachy whips back around and checks "
        "the camera for movement, returning caught=true if the user was still moving. Narrate the "
        "result yourself (call them out if caught, praise them if still). Repeat green_light/red_light "
        "rounds until the user asks to stop, then call move_head with direction='front' to face them "
        "again. Requires the camera; do not use if the camera is disabled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["green_light", "red_light"],
                "description": "'green_light' to turn away, 'red_light' to turn back fast and check for movement.",
            },
        },
        "required": ["phase"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run the requested phase of the game."""
        phase_raw = kwargs.get("phase")
        if phase_raw not in ("green_light", "red_light"):
            return {"error": "phase must be 'green_light' or 'red_light'"}
        phase: Phase = phase_raw
        logger.info("Tool call: red_light_green_light phase=%s", phase)

        try:
            if phase == "green_light":
                return self._look_away(deps)
            return await self._turn_and_check(deps)
        except Exception as e:
            logger.error("red_light_green_light failed")
            return {"error": f"red_light_green_light failed: {type(e).__name__}: {e}"}

    def _look_away(self, deps: ToolDependencies) -> Dict[str, Any]:
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

        return {"status": "looking away, green light"}

    async def _turn_and_check(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Turn back to face the user fast, then check the camera for movement."""
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

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
            return {"error": "No frame available"}

        max_diff = max(float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames)))
        caught = max_diff > _MOTION_DIFF_THRESHOLD

        return {"status": "red light", "caught": caught}
