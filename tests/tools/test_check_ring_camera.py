"""Tests for the Ring camera snapshot tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.ring_client import RingNotConfiguredError, RingDeviceNotFoundError
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.check_ring_camera import RingCamera


def _deps(ring_client: object | None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        ring_client=ring_client,
    )


@pytest.mark.asyncio
async def test_returns_error_when_ring_not_configured() -> None:
    """No ring_client at all -> error, no crash."""
    result = await RingCamera()(_deps(None), location="garden")

    assert "error" in result


@pytest.mark.asyncio
async def test_returns_single_snapshot_for_named_location() -> None:
    """A single known location returns one image entry."""
    ring_client = MagicMock()
    ring_client.async_get_device_snapshot = AsyncMock(return_value=b"\xff\xd8jpeg\xff\xd9")

    result = await RingCamera()(_deps(ring_client), location="Garden")

    assert result["images"][0]["label"] == "Garden"
    assert "b64_im" in result["images"][0]
    ring_client.async_get_device_snapshot.assert_awaited_once_with("Garden")


@pytest.mark.asyncio
async def test_returns_all_snapshots_for_all_location() -> None:
    """'all' fans out to every configured camera."""
    ring_client = MagicMock()
    ring_client.async_list_locations = AsyncMock(return_value=["Garden", "Front Door", "bod"])
    ring_client.async_get_device_snapshot = AsyncMock(return_value=b"\xff\xd8jpeg\xff\xd9")

    result = await RingCamera()(_deps(ring_client), location="all")

    labels = {image["label"] for image in result["images"]}
    assert labels == {"Garden", "Front Door", "bod"}
    assert ring_client.async_get_device_snapshot.await_count == 3


@pytest.mark.asyncio
async def test_one_failing_camera_does_not_block_the_others() -> None:
    """A single camera erroring out still returns the working ones."""
    ring_client = MagicMock()
    ring_client.async_list_locations = AsyncMock(return_value=["Garden", "bod"])

    async def snapshot(location: str) -> bytes:
        if location == "bod":
            raise RuntimeError("camera offline")
        return b"\xff\xd8jpeg\xff\xd9"

    ring_client.async_get_device_snapshot = AsyncMock(side_effect=snapshot)

    result = await RingCamera()(_deps(ring_client), location="all")

    by_label = {image["label"]: image for image in result["images"]}
    assert "b64_im" in by_label["Garden"]
    assert "error" in by_label["bod"]


@pytest.mark.asyncio
async def test_unknown_device_name_reports_error() -> None:
    """A location that doesn't match any Ring device name returns an error entry."""
    ring_client = MagicMock()
    ring_client.async_get_device_snapshot = AsyncMock(
        side_effect=RingDeviceNotFoundError("No Ring device named 'attic'."),
    )

    result = await RingCamera()(_deps(ring_client), location="attic")

    assert "error" in result["images"][0]


@pytest.mark.asyncio
async def test_missing_token_cache_reports_error() -> None:
    """No cached Ring login yet -> per-camera error, not a crash."""
    ring_client = MagicMock()
    ring_client.async_get_device_snapshot = AsyncMock(
        side_effect=RingNotConfiguredError("No cached Ring login found."),
    )

    result = await RingCamera()(_deps(ring_client), location="garden")

    assert "error" in result["images"][0]


@pytest.mark.asyncio
async def test_empty_location_is_rejected() -> None:
    """An empty location string is rejected before touching the ring client."""
    result = await RingCamera()(_deps(MagicMock()), location="  ")

    assert "error" in result
