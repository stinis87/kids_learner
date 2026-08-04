"""Tests for the talk_to_door tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.talk_to_door import TalkToDoor


def _deps(start_door_call: object | None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        start_door_call=start_door_call,
    )


@pytest.mark.asyncio
async def test_returns_error_when_door_calls_unavailable() -> None:
    """No start_door_call callback wired -> error, no crash."""
    result = await TalkToDoor()(_deps(None), location="Front Door")

    assert "error" in result


@pytest.mark.asyncio
async def test_delegates_to_start_door_call_with_location() -> None:
    """A valid location is forwarded to the wired start_door_call callback."""
    start_door_call = AsyncMock(return_value={"status": "connected", "location": "Front Door"})

    result = await TalkToDoor()(_deps(start_door_call), location="Front Door")

    assert result == {"status": "connected", "location": "Front Door"}
    start_door_call.assert_awaited_once_with("Front Door")


@pytest.mark.asyncio
async def test_empty_location_is_rejected() -> None:
    """An empty location string is rejected before calling start_door_call."""
    start_door_call = AsyncMock()

    result = await TalkToDoor()(_deps(start_door_call), location="   ")

    assert "error" in result
    start_door_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_propagates_error_from_start_door_call() -> None:
    """An error dict from start_door_call (e.g. unknown device) is returned unchanged."""
    start_door_call = AsyncMock(return_value={"error": "No Ring device named 'attic'."})

    result = await TalkToDoor()(_deps(start_door_call), location="attic")

    assert result == {"error": "No Ring device named 'attic'."}
