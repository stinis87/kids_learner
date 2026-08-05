"""Tests for blocking sensitive tools while a live doorbell call is active.

Doorbell call audio is fed into the same conversation as the room mic (see
huggingface_realtime.py's _pump_door_call_audio_in), so a visitor at the door
has the same voice access as the person in the room. A handful of sensitive
tools must stay unreachable to anyone while a call is open.
"""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import (
    DOOR_CALL_BLOCKED_TOOLS,
    ToolDependencies,
    dispatch_tool_call,
)


@pytest.mark.asyncio
async def test_blocked_tool_is_rejected_while_a_door_call_is_active() -> None:
    """A sensitive tool call is refused while is_door_call_active() reports True."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=MagicMock(return_value={"status": "sleeping"}),
        is_door_call_active=lambda: True,
    )

    result = await dispatch_tool_call("go_to_sleep", "{}", deps)

    assert "error" in result


@pytest.mark.asyncio
async def test_check_ring_camera_is_rejected_while_a_door_call_is_active() -> None:
    """Ring camera snapshots are blocked during a call too, not just the memory/sleep/profile tools."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        ring_client=MagicMock(),
        is_door_call_active=lambda: True,
    )

    result = await dispatch_tool_call("check_ring_camera", "{}", deps)

    assert "error" in result
    assert "doorbell call" in result["error"]


@pytest.mark.asyncio
async def test_blocked_tool_runs_normally_when_no_call_is_active() -> None:
    """The same tool runs as usual once the call is over (is_door_call_active() is False)."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=MagicMock(return_value={"status": "sleeping"}),
        is_door_call_active=lambda: False,
    )

    result = await dispatch_tool_call("go_to_sleep", "{}", deps)

    assert result == {"status": "sleeping"}


@pytest.mark.asyncio
async def test_blocked_tool_runs_normally_without_the_callback_wired() -> None:
    """Runtimes that never wire is_door_call_active (e.g. no Ring configured) are unaffected."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=MagicMock(return_value={"status": "sleeping"}),
    )

    result = await dispatch_tool_call("go_to_sleep", "{}", deps)

    assert result == {"status": "sleeping"}


@pytest.mark.asyncio
async def test_unblocked_tool_is_unaffected_by_an_active_door_call() -> None:
    """Tools outside the blocklist (e.g. end_door_call) keep working during a call."""
    end_door_call = MagicMock(return_value={"status": "ended"})

    async def _end_door_call() -> dict[str, object]:
        return end_door_call()

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        end_door_call=_end_door_call,
        is_door_call_active=lambda: True,
    )

    result = await dispatch_tool_call("end_door_call", "{}", deps)

    assert result == {"status": "ended"}


def test_door_call_blocked_tools_cover_the_sensitive_ones() -> None:
    """Guard against silently dropping a tool from the blocklist during a refactor."""
    assert DOOR_CALL_BLOCKED_TOOLS == frozenset(
        {"remember", "forget", "go_to_sleep", "switch_profile", "check_ring_camera"}
    )
