from __future__ import annotations
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import ClassVar, TypeAlias
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.streaming import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from reachy_mini_conversation_app.wake_word import WakeWordGate
from reachy_mini_conversation_app.idle_policy import start_idle_tool_call
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs

# Antenna poses double as a facial expression for the wake-word gate: relaxed/tucked while
# dormant, perked up and alert once the wake word opens the gate.
_ANTENNAS_DORMANT: tuple[float, float] = (0.0, 0.0)
_ANTENNAS_ALERT: tuple[float, float] = (0.6, 0.6)
_WAKE_WORD_HIDE_DURATION_S = 0.5  # time to tuck the head down, matches red_light_green_light's hide snap
_WAKE_WORD_POP_UP_DURATION_S = 0.35  # quicker snap back up than hiding


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract and idle behavior for realtime conversation backends."""

    IDLE_BEHAVIOR_THRESHOLD_S: ClassVar[float] = 180.0

    deps: ToolDependencies
    tool_manager: BackgroundToolManager
    output_queue: asyncio.Queue[QueueItem]
    last_activity_time: float
    last_idle_behavior_time: float
    _activity_observer: Callable[[str], None] | None = None

    def __init__(self) -> None:
        """Initialize the stream handler and shared idle/activity tracking."""
        super().__init__()
        self.last_activity_time = time.monotonic()
        self.last_idle_behavior_time = self.last_activity_time
        self._wake_word_gate = WakeWordGate(
            enabled=config.WAKE_WORD_ENABLED,
            model_name=config.WAKE_WORD_MODEL,
            threshold=config.WAKE_WORD_THRESHOLD,
        )

    def set_activity_observer(self, observer: Callable[[str], None] | None) -> None:
        """Attach or detach an activity observer. Pass None to clear."""
        self._activity_observer = observer

    def _mark_activity(self, reason: str) -> None:
        """Record non-idle conversation activity for the idle timer."""
        self.last_activity_time = time.monotonic()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)
        if self._activity_observer is not None:
            try:
                self._activity_observer(reason)
            except Exception:
                logger.debug("activity observer raised (ignored)", exc_info=True)

    def _idle_behavior_ready(self) -> bool:
        """Return whether idle behavior may run now. Backends can add guards."""
        return True

    async def _gate_audio_frame(self, sample_rate: int, audio_frame: NDArray[np.int16]) -> bool:
        """Run the wake-word gate on a mic frame; return whether it should be forwarded downstream."""
        was_active = self._wake_word_gate.active
        self._wake_word_gate.process(sample_rate, audio_frame)
        is_active = self._wake_word_gate.active
        if is_active != was_active:
            self._queue_wake_word_animation(is_active)
            if not is_active:
                await self._clear_pending_input_audio()
        return is_active

    def _queue_wake_word_animation(self, active: bool) -> None:
        """Snap the head to alert when the gate opens, or tuck it down when it closes."""
        try:
            current_head_pose = self.deps.reachy_mini.get_current_head_pose()
            head_joints, current_antennas = self.deps.reachy_mini.get_current_joint_positions()
            current_body_yaw = head_joints[0]
        except Exception as e:
            logger.warning("Could not read robot pose for wake-word animation (%s); skipping.", e)
            return

        if active:
            target_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
            target_antennas = _ANTENNAS_ALERT
            duration = _WAKE_WORD_POP_UP_DURATION_S
        else:
            target_head_pose = SLEEP_HEAD_POSE.astype(np.float32)
            target_antennas = _ANTENNAS_DORMANT
            duration = _WAKE_WORD_HIDE_DURATION_S

        move = GotoQueueMove(
            target_head_pose=target_head_pose,
            start_head_pose=current_head_pose,
            target_antennas=target_antennas,
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=current_body_yaw,
            start_body_yaw=current_body_yaw,
            duration=duration,
        )
        self.deps.movement_manager.queue_move(move)
        self.deps.movement_manager.set_moving_state(duration)

    async def _clear_pending_input_audio(self) -> None:
        """Discard buffered realtime input audio on gate deactivation. No-op unless overridden."""
        return None

    async def emit(self) -> HandlerOutput:
        """Emit the next queued output, triggering local idle behavior when due."""
        now = time.monotonic()
        idle_duration = now - self.last_activity_time
        idle_behavior_duration = now - self.last_idle_behavior_time
        if (
            idle_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and idle_behavior_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and self._idle_behavior_ready()
            and self.deps.movement_manager.is_idle()
        ):
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle tool skipped (connection closed?): %s", e)
                return None
            self.last_idle_behavior_time = now
        handler_output = await wait_for_item(self.output_queue)
        return handler_output

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Run a locally selected idle tool without sending an idle turn to the model."""
        if not self._is_connected():
            logger.debug("No active session; cannot run idle tool")
            return

        available_tool_names = {spec["name"] for spec in get_tool_specs()}
        await start_idle_tool_call(
            deps=self.deps,
            tool_manager=self.tool_manager,
            output_queue=self.output_queue,
            available_tool_names=available_tool_names,
            idle_duration=idle_duration,
        )

    @abstractmethod
    def _is_connected(self) -> bool:
        """Return whether the backend session/connection is currently open."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...
