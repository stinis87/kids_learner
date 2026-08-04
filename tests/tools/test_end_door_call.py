"""Tests for the end_door_call tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.end_door_call import EndDoorCall


def _deps(end_door_call: object | None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        end_door_call=end_door_call,
    )


@pytest.mark.asyncio
async def test_returns_error_when_door_calls_unavailable() -> None:
    """No end_door_call callback wired -> error, no crash."""
    result = await EndDoorCall()(_deps(None))

    assert "error" in result


@pytest.mark.asyncio
async def test_delegates_to_end_door_call() -> None:
    """Calling the tool hangs up via the wired end_door_call callback."""
    end_door_call = AsyncMock(return_value={"status": "ended", "location": "Front Door"})

    result = await EndDoorCall()(_deps(end_door_call))

    assert result == {"status": "ended", "location": "Front Door"}
    end_door_call.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_propagates_error_when_no_active_call() -> None:
    """An error dict (e.g. no active call) is returned unchanged."""
    end_door_call = AsyncMock(return_value={"error": "No active doorbell call."})

    result = await EndDoorCall()(_deps(end_door_call))

    assert result == {"error": "No active doorbell call."}
