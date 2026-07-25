import random
import asyncio
import logging
from typing import Any, Dict, List, Tuple, Literal

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.tools.people_detector import PeopleDetector, ensure_pose_model_downloaded


logger = logging.getLogger(__name__)

Phase = Literal["green_light", "red_light"]

# Timings and thresholds mirror the "expert mode" of this game
# (ChristmasTree/expert-red-light-green-light): quick head-hide/reveal snaps, antennas
# that act alert or coy, occasional fake-out green lights, and a long, randomized watch
# window rather than a fixed one, so the pacing itself keeps players honest.
_HIDE_DURATION_S = 0.08
_REVEAL_DURATION_S = 0.08
_RED_SETTLE_S = 0.3  # pause after snapping up before scanning, to let motion blur clear
_GREEN_LIGHT_MIN_S = 0.8
_GREEN_LIGHT_MAX_S = 2.4
_FAKEOUT_PROB = 0.08
_FAKEOUT_GREEN_S = 0.45
_RED_LIGHT_MIN_S = 3.0
_RED_LIGHT_MAX_S = 7.0
_MOTION_SAMPLE_INTERVAL_S = 0.12
_POSE_BUFFER_LEN = 5
# Per-person pose-landmark position std-dev across the buffered frames; above this,
# a person is flagged as moving. Matches pollen-robotics/red_light_green_light.
_MOTION_STD_THRESHOLD = 3.0
_CAUGHT_TURN_DURATION_S = 0.3
_CAUGHT_YAW_RANGE_DEG = (-80.0, 80.0)

# Antenna poses double as Reachy's facial expression here: relaxed while hiding its
# eyes, perked up and alert while watching, cocked sideways when calling someone out.
_ANTENNAS_HIDDEN: Tuple[float, float] = (0.0, 0.0)
_ANTENNAS_WATCHING: Tuple[float, float] = (0.6, 0.6)
_ANTENNAS_CALLING_OUT: Tuple[float, float] = (0.5, -0.5)


class RedLightGreenLight(Tool):
    """Play one phase of Red Light, Green Light with the user."""

    name = "red_light_green_light"
    description = (
        "Play Red Light, Green Light, one phase per call. Step 1: say 'Green light!' out loud AS you "
        "call this tool with phase='green_light' — Reachy tucks its head down to hide its eyes (they may "
        "move now) for a short randomized moment (usually under 2.5 seconds, occasionally a lightning-fast "
        "fake-out) that it decides itself. Step 2: the instant that call returns, without waiting for the "
        "user to say or do anything, say 'Red light!' and immediately call this tool again with "
        "phase='red_light' — Reachy snaps its head up and watches the camera for several seconds, and if "
        "it catches someone moving it immediately turns to look right at them, returning caught=true. "
        "Narrate that result right away (call them out if caught, praise them if still). Repeat this "
        "green_light then red_light sequence for as many rounds as the user wants, always chaining "
        "red_light right after green_light on your own. When the user asks to stop, call move_head with "
        "direction='front' to face them again. Requires the camera; do not use if the camera is disabled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["green_light", "red_light"],
                "description": (
                    "'green_light' to hide its eyes and wait a randomized moment; "
                    "'red_light' to snap back up and watch for movement."
                ),
            },
        },
        "required": ["phase"],
    }
    _detector: "PeopleDetector | None" = None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run the requested phase of the game."""
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

        phase_raw = kwargs.get("phase")
        if phase_raw not in ("green_light", "red_light"):
            return {"error": "phase must be 'green_light' or 'red_light'"}
        phase: Phase = phase_raw
        logger.info("Tool call: red_light_green_light phase=%s", phase)

        try:
            if phase == "green_light":
                return await self._hide_and_wait(deps)
            return await self._watch_for_movement(deps)
        except Exception as e:
            logger.error("red_light_green_light failed")
            return {"error": f"red_light_green_light failed: {type(e).__name__}: {e}"}

    async def _hide_and_wait(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Tuck the head down to hide its eyes, then wait a randomized short moment."""
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]

        move = GotoQueueMove(
            target_head_pose=SLEEP_HEAD_POSE.astype(np.float32),
            start_head_pose=current_head_pose,
            target_antennas=_ANTENNAS_HIDDEN,
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=0,
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
            "status": "time is up, say 'Red light!' and call phase='red_light' now",
            "green_light_seconds": round(green_light_duration, 2),
            "fakeout": is_fakeout,
        }

    async def _watch_for_movement(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Snap the head back up, watch for movement, and turn to look at whoever gets caught."""
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

        detector = await self._get_detector()
        caught_center, frame_width, frames_read = await self._scan_for_movement(deps, detector)

        if frames_read < 2:
            raise RuntimeError("No frame available")

        caught = caught_center is not None
        if caught_center is not None and frame_width is not None:
            await self._turn_to_caught_player(deps, caught_center, frame_width)

        return {"status": "red light", "caught": caught}

    async def _scan_for_movement(
        self, deps: ToolDependencies, detector: PeopleDetector
    ) -> tuple[Tuple[int, int] | None, int | None, int]:
        """Watch the camera and return the pixel position of the first person caught moving, if any.

        Buffers each detected person's upper-body landmarks across the last few frames and flags
        movement by the std-dev of their positions (like pollen-robotics/red_light_green_light).
        Stops as soon as someone is caught instead of always running the full scan window, matching
        how the expert-mode game snaps to a mover the instant they slip.
        """
        scan_for = random.uniform(_RED_LIGHT_MIN_S, _RED_LIGHT_MAX_S)
        person_buffers: List[List[List[Tuple[int, int]]]] = []
        frames_read = 0
        frame_width: int | None = None
        elapsed = 0.0

        while elapsed < scan_for:
            frame = await asyncio.to_thread(deps.reachy_mini.media.get_frame)
            if frame is not None:
                frames_read += 1
                frame_width = frame.shape[1]
                people = await asyncio.to_thread(detector.detect, frame)
                while len(person_buffers) < len(people):
                    person_buffers.append([])
                for person_index, landmarks in enumerate(people):
                    buffer = person_buffers[person_index]
                    buffer.append(landmarks)
                    del buffer[:-_POSE_BUFFER_LEN]
                    if len(buffer) >= _POSE_BUFFER_LEN:
                        score = float(np.std(buffer, axis=0).mean())
                        if score > _MOTION_STD_THRESHOLD:
                            return landmarks[0], frame_width, frames_read

            await asyncio.sleep(_MOTION_SAMPLE_INTERVAL_S)
            elapsed += _MOTION_SAMPLE_INTERVAL_S

        return None, frame_width, frames_read

    async def _turn_to_caught_player(
        self, deps: ToolDependencies, shoulder_center: Tuple[int, int], frame_width: int
    ) -> None:
        """Turn the head to look at the pixel position of whoever just got caught moving."""
        normalized_x = ((shoulder_center[0] / frame_width) - 0.5) * 2
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

    async def _get_detector(self) -> PeopleDetector:
        """Lazily build the pose detector, downloading its model on first use."""
        if self._detector is None:
            model_path = await asyncio.to_thread(ensure_pose_model_downloaded)
            self._detector = await asyncio.to_thread(PeopleDetector, model_path)
        return self._detector
