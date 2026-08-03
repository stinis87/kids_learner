"""Tests for the Ring history query tool."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.ring_client import (
    RingEvent,
    RingHistorySummary,
    RingEventNotFoundError,
    RingNoEventsFoundError,
    RingNotConfiguredError,
    RingDeviceNotFoundError,
    RingDayNotRecognizedError,
    RingRecordingUnavailableError,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.check_ring_history import RingHistory


def _deps(ring_client: object | None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        ring_client=ring_client,
    )


def _event(kind: str = "motion", event_id: int = 1) -> RingEvent:
    return RingEvent(
        device_name="Garden", event_id=event_id, kind=kind, created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )


@pytest.mark.asyncio
async def test_returns_error_when_ring_not_configured() -> None:
    """No ring_client at all -> error, no crash."""
    result = await RingHistory()(_deps(None), location="garden", day="today")

    assert "error" in result


@pytest.mark.asyncio
async def test_empty_location_is_rejected() -> None:
    """An empty location string is rejected before touching the ring client."""
    result = await RingHistory()(_deps(MagicMock()), location="  ", day="today")

    assert "error" in result


@pytest.mark.asyncio
async def test_empty_day_is_rejected() -> None:
    """An empty day string is rejected before touching the ring client."""
    result = await RingHistory()(_deps(MagicMock()), location="garden", day="  ")

    assert "error" in result


@pytest.mark.asyncio
async def test_returns_summary_without_images_by_default() -> None:
    """A plain history query returns counts/times only, no frames fetched."""
    ring_client = MagicMock()
    summary = RingHistorySummary(
        device_name="Garden", day=date(2024, 1, 1), events=[_event("motion", 1), _event("ding", 2)]
    )
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)

    result = await RingHistory()(_deps(ring_client), location="garden", day="today")

    assert result["event_count"] == 2
    assert "images" not in result
    ring_client.async_describe_event.assert_not_called()


@pytest.mark.asyncio
async def test_no_events_reports_zero_count() -> None:
    """A quiet day reports zero events rather than an error."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)

    result = await RingHistory()(_deps(ring_client), location="garden", day="yesterday")

    assert result["event_count"] == 0
    assert result["events"] == []


@pytest.mark.asyncio
async def test_unknown_location_reports_error() -> None:
    """A location that doesn't match any Ring device name returns an error."""
    ring_client = MagicMock()
    ring_client.async_get_history_for_day = AsyncMock(
        side_effect=RingDeviceNotFoundError("No Ring device named 'attic'.")
    )

    result = await RingHistory()(_deps(ring_client), location="attic", day="today")

    assert "error" in result


@pytest.mark.asyncio
async def test_unrecognized_day_reports_error() -> None:
    """A day string that isn't 'today'/'yesterday'/an ISO date returns an error."""
    ring_client = MagicMock()
    ring_client.async_get_history_for_day = AsyncMock(
        side_effect=RingDayNotRecognizedError("Could not understand day.")
    )

    result = await RingHistory()(_deps(ring_client), location="garden", day="next tuesday")

    assert "error" in result


@pytest.mark.asyncio
async def test_missing_token_cache_reports_error() -> None:
    """No cached Ring login yet -> error, not a crash."""
    ring_client = MagicMock()
    ring_client.async_get_history_for_day = AsyncMock(
        side_effect=RingNotConfiguredError("No cached Ring login found.")
    )

    result = await RingHistory()(_deps(ring_client), location="garden", day="today")

    assert "error" in result


@pytest.mark.asyncio
async def test_describe_event_attaches_frames_when_requested() -> None:
    """describe_event fetches and attaches frames from the selected event."""
    ring_client = MagicMock()
    summary = RingHistorySummary(
        device_name="Garden", day=date(2024, 1, 1), events=[_event("motion", 2), _event("ding", 1)]
    )
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock(return_value=(_event("motion", 2), [b"frame-one", b"frame-two"]))

    result = await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="latest")

    assert len(result["images"]) == 2
    assert all("b64_im" in image for image in result["images"])
    ring_client.async_describe_event.assert_awaited_once_with("garden", "today", "latest")


@pytest.mark.asyncio
async def test_describe_event_passes_through_ordinal_and_time_selectors() -> None:
    """A user-picked selector like 'second' or '2pm' is forwarded to the client unchanged."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[_event()])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock(return_value=(_event(), [b"frame"]))

    await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="second")
    ring_client.async_describe_event.assert_awaited_once_with("garden", "today", "second")

    ring_client.async_describe_event.reset_mock()
    await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="2pm")
    ring_client.async_describe_event.assert_awaited_once_with("garden", "today", "2pm")


@pytest.mark.asyncio
async def test_describe_event_not_called_when_no_events() -> None:
    """Describing is skipped entirely when there are no events for the day, even if requested."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock()

    result = await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="latest")

    assert "images" not in result
    ring_client.async_describe_event.assert_not_called()


@pytest.mark.asyncio
async def test_describe_event_surfaces_no_subscription_error() -> None:
    """A missing Ring Protect subscription is reported clearly rather than silently omitted."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[_event()])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock(
        side_effect=RingRecordingUnavailableError("'Garden' has no active Ring Protect subscription.")
    )

    result = await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="latest")

    assert "images" not in result
    assert "Ring Protect" in result["description_error"]


@pytest.mark.asyncio
async def test_describe_event_surfaces_no_events_found_error() -> None:
    """A day with no matching event history reports the error rather than crashing."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[_event()])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock(side_effect=RingNoEventsFoundError("No events found."))

    result = await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="latest")

    assert "images" not in result
    assert "description_error" in result


@pytest.mark.asyncio
async def test_describe_event_surfaces_unmatched_selector_error() -> None:
    """A selector that doesn't match any event (e.g. out-of-range ordinal) reports the error."""
    ring_client = MagicMock()
    summary = RingHistorySummary(device_name="Garden", day=date(2024, 1, 1), events=[_event()])
    ring_client.async_get_history_for_day = AsyncMock(return_value=summary)
    ring_client.async_describe_event = AsyncMock(
        side_effect=RingEventNotFoundError("There were only 1 event(s) that day, no 'fifth' one.")
    )

    result = await RingHistory()(_deps(ring_client), location="garden", day="today", describe_event="fifth")

    assert "images" not in result
    assert "description_error" in result
