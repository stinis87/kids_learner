"""Tests for the Ring client's location alias resolution and device lookup."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ring_doorbell import RingError

from reachy_mini_conversation_app import ring_client as ring_client_module
from reachy_mini_conversation_app.ring_client import (
    RingClient,
    RingEventNotFoundError,
    RingNoEventsFoundError,
    RingDeviceNotFoundError,
    RingDayNotRecognizedError,
    RingRecordingUnavailableError,
)


def _fake_device(
    name: str,
    device_id: int = 123,
    history: list[dict[str, object]] | None = None,
    timezone_name: str = "UTC",
    has_subscription: bool = True,
) -> MagicMock:
    device = MagicMock()
    device.name = name
    device.id = device_id
    device.timezone = timezone_name
    device.has_subscription = has_subscription
    device.async_history = AsyncMock(return_value=history or [])
    device.async_recording_download = AsyncMock(return_value=b"fake-mp4-bytes")
    device.async_recording_url = AsyncMock(return_value=None)
    return device


def _history_entry(event_id: int, kind: str, created_at: datetime | None = None) -> dict[str, object]:
    return {
        "id": event_id,
        "kind": kind,
        "created_at": created_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
    }


def _fake_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.content = content
    return response


def _client_with_devices(devices: list[MagicMock], jpeg_bytes: bytes = b"\xff\xd8jpeg\xff\xd9") -> RingClient:
    client = RingClient()
    ring = MagicMock()
    ring.async_update_devices = AsyncMock()
    ring.video_devices = MagicMock(return_value=devices)
    ring.async_query = AsyncMock(return_value=_fake_response(jpeg_bytes))
    client._get_ring = AsyncMock(return_value=ring)
    return client


@pytest.mark.asyncio
async def test_exact_device_name_matches_case_insensitively() -> None:
    """A location matching a device name directly (any case) resolves to that device."""
    client = _client_with_devices([_fake_device("Garden"), _fake_device("Front Door")])

    await client.async_get_device_snapshot("garden")

    client._get_ring.return_value.async_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_norwegian_alias_hage_resolves_to_garden() -> None:
    """'Hage' is a Norwegian synonym for the 'Garden' device."""
    garden = _fake_device("Garden", device_id=1)
    client = _client_with_devices([garden, _fake_device("Front Door", 2), _fake_device("bod", 3)])

    await client.async_get_device_snapshot("Hage")

    url = client._get_ring.return_value.async_query.await_args.args[0]
    assert url == ring_client_module._SNAPSHOT_ENDPOINT.format(1)


@pytest.mark.asyncio
async def test_norwegian_alias_framsiden_resolves_to_front_door() -> None:
    """'Framsiden' is a Norwegian synonym for the 'Front Door' device."""
    front_door = _fake_device("Front Door", device_id=2)
    client = _client_with_devices([_fake_device("Garden", 1), front_door, _fake_device("bod", 3)])

    await client.async_get_device_snapshot("framsiden")

    url = client._get_ring.return_value.async_query.await_args.args[0]
    assert url == ring_client_module._SNAPSHOT_ENDPOINT.format(2)


@pytest.mark.asyncio
async def test_unknown_location_raises_with_known_device_names() -> None:
    """An unmatched location raises, listing the actually configured device names."""
    client = _client_with_devices([_fake_device("Garden"), _fake_device("bod")])

    with pytest.raises(RingDeviceNotFoundError, match="Garden.*bod|bod.*Garden"):
        await client.async_get_device_snapshot("attic")


@pytest.mark.asyncio
async def test_snapshot_retries_after_transient_ring_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that briefly times out on the snapshot endpoint still returns an image."""
    monkeypatch.setattr(ring_client_module, "_SNAPSHOT_ATTEMPT_BACKOFF_S", 0)
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    garden = _fake_device("Garden")
    client = _client_with_devices([garden])
    client._get_ring.return_value.async_query = AsyncMock(
        side_effect=[RingError("404 server timeout"), _fake_response(jpeg_bytes)]
    )

    result = await client.async_get_device_snapshot("garden")

    assert result == jpeg_bytes
    assert client._get_ring.return_value.async_query.await_count == 2


@pytest.mark.asyncio
async def test_snapshot_gives_up_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that never produces a snapshot raises a clear, actionable error."""
    monkeypatch.setattr(ring_client_module, "_SNAPSHOT_ATTEMPT_BACKOFF_S", 0)
    garden = _fake_device("Garden")
    client = _client_with_devices([garden])
    client._get_ring.return_value.async_query = AsyncMock(side_effect=RingError("404 server timeout"))

    with pytest.raises(RuntimeError, match="Garden.*did not return a snapshot"):
        await client.async_get_device_snapshot("garden")

    assert client._get_ring.return_value.async_query.await_count == ring_client_module._SNAPSHOT_ATTEMPTS


@pytest.mark.asyncio
async def test_latest_events_returns_most_recent_matching_kind_per_device() -> None:
    """Each device's newest history entry matching the requested kinds is returned."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(2, "motion"), _history_entry(1, "on_demand")],
    )
    front_door = _fake_device(
        "Front Door",
        history=[_history_entry(5, "ding"), _history_entry(4, "motion")],
    )
    client = _client_with_devices([garden, front_door])

    events = await client.async_get_latest_events(("motion", "ding"))

    assert events["Garden"].event_id == 2
    assert events["Garden"].kind == "motion"
    assert events["Front Door"].event_id == 5
    assert events["Front Door"].kind == "ding"


@pytest.mark.asyncio
async def test_latest_events_skips_unmatched_kinds() -> None:
    """A device whose most recent entries are all outside the requested kinds is omitted."""
    bod = _fake_device("Bod", history=[_history_entry(9, "on_demand")])
    client = _client_with_devices([bod])

    events = await client.async_get_latest_events(("motion", "ding"))

    assert "Bod" not in events


@pytest.mark.asyncio
async def test_latest_events_omits_device_with_no_history() -> None:
    """A device with an empty history simply doesn't appear in the result."""
    bod = _fake_device("Bod", history=[])
    client = _client_with_devices([bod])

    events = await client.async_get_latest_events(("motion", "ding"))

    assert events == {}


@pytest.mark.asyncio
async def test_latest_events_skips_device_on_history_error() -> None:
    """One device's history fetch failing doesn't prevent checking the others."""
    broken = _fake_device("Garden")
    broken.async_history = AsyncMock(side_effect=RingError("boom"))
    healthy = _fake_device("Front Door", history=[_history_entry(3, "ding")])
    client = _client_with_devices([broken, healthy])

    events = await client.async_get_latest_events(("motion", "ding"))

    assert "Garden" not in events


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("today", date(2024, 6, 15)),
        ("Yesterday", date(2024, 6, 14)),
        ("day before yesterday", date(2024, 6, 13)),
        ("2024-01-05", date(2024, 1, 5)),
        ("i dag", date(2024, 6, 15)),
        ("I går", date(2024, 6, 14)),
        ("i forgårs", date(2024, 6, 13)),
    ],
)
def test_resolve_day_handles_relative_and_iso_dates(day: str, expected: date) -> None:
    """'today'/'yesterday'/'day before yesterday' (English and Norwegian) resolve relative to now; ISO dates resolve directly."""
    tz = ring_client_module.ZoneInfo("UTC")
    with patch(
        "reachy_mini_conversation_app.ring_client.datetime",
        wraps=datetime,
    ) as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        resolved = ring_client_module.resolve_day(day, tz)

    assert resolved == expected


def test_resolve_day_rejects_unrecognized_string() -> None:
    """A day string that isn't relative or ISO-formatted raises a clear error."""
    with pytest.raises(RingDayNotRecognizedError, match="next tuesday"):
        ring_client_module.resolve_day("next tuesday", ring_client_module.ZoneInfo("UTC"))


@pytest.mark.asyncio
async def test_history_for_day_filters_by_window_and_kind() -> None:
    """Only watched-kind events within the requested day's window are returned."""
    garden = _fake_device(
        "Garden",
        history=[
            _history_entry(3, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc)),
            _history_entry(2, "on_demand", datetime(2024, 1, 1, 9, tzinfo=timezone.utc)),
            _history_entry(1, "ding", datetime(2023, 12, 31, 23, tzinfo=timezone.utc)),
        ],
    )
    client = _client_with_devices([garden])

    summary = await client.async_get_history_for_day("garden", "2024-01-01")

    assert summary.day == date(2024, 1, 1)
    assert [event.event_id for event in summary.events] == [3]


@pytest.mark.asyncio
async def test_history_for_day_pages_until_window_start_is_reached() -> None:
    """A day with more events than one page requires paging with the `older_than` cursor."""
    garden = _fake_device("Garden")
    page_size = ring_client_module._HISTORY_DAY_PAGE_SIZE
    first_page = [
        _history_entry(page_size - i, "motion", datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        for i in range(page_size)
    ]
    second_page = [_history_entry(1, "motion", datetime(2023, 12, 31, 12, tzinfo=timezone.utc))]
    garden.async_history = AsyncMock(side_effect=[first_page, second_page])
    client = _client_with_devices([garden])

    summary = await client.async_get_history_for_day("garden", "2024-01-01")

    assert len(summary.events) == page_size
    assert garden.async_history.await_count == 2
    second_call_kwargs = garden.async_history.await_args_list[1].kwargs
    assert second_call_kwargs["older_than"] == first_page[-1]["id"]


@pytest.mark.asyncio
async def test_describe_event_downloads_and_extracts_frames_for_latest() -> None:
    """The 'latest' selector (the default) downloads the most recent event's clip."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(2, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    client = _client_with_devices([garden])

    with patch(
        "reachy_mini_conversation_app.ring_client.async_extract_evenly_spaced_frames",
        AsyncMock(return_value=[b"frame1", b"frame2"]),
    ) as mock_extract:
        event, frames = await client.async_describe_event("garden", "2024-01-01", "latest")

    assert event.event_id == 2
    assert frames == [b"frame1", b"frame2"]
    garden.async_recording_download.assert_awaited_once()
    mock_extract.assert_awaited_once_with(b"fake-mp4-bytes", ring_client_module._DESCRIBE_FRAME_COUNT)


@pytest.mark.asyncio
async def test_describe_event_falls_back_to_share_url_on_404() -> None:
    """A 404 from the primary download endpoint falls back to the CDN share URL."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(2, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    garden.async_recording_download = AsyncMock(
        side_effect=RingError("HTTP error with status code 404 during query of url ...: 404, message='Not Found'")
    )
    garden.async_recording_url = AsyncMock(return_value="https://cdn.ring.com/signed-clip.mp4")
    client = _client_with_devices([garden])

    fake_response = MagicMock()
    fake_response.content = b"fake-mp4-bytes"
    fake_response.raise_for_status = MagicMock()
    fake_http_client = AsyncMock()
    fake_http_client.get = AsyncMock(return_value=fake_response)
    fake_http_client.__aenter__ = AsyncMock(return_value=fake_http_client)
    fake_http_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "reachy_mini_conversation_app.ring_client.async_extract_evenly_spaced_frames",
            AsyncMock(return_value=[b"frame1"]),
        ),
        patch("httpx.AsyncClient", return_value=fake_http_client),
    ):
        event, frames = await client.async_describe_event("garden", "2024-01-01", "latest")

    assert event.event_id == 2
    assert frames == [b"frame1"]
    garden.async_recording_url.assert_awaited_once_with(2)
    fake_http_client.get.assert_awaited_once_with("https://cdn.ring.com/signed-clip.mp4")


@pytest.mark.asyncio
async def test_describe_event_raises_when_share_url_fallback_also_unavailable() -> None:
    """A 404 with no share URL available surfaces as a recording-unavailable error."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(2, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    garden.async_recording_download = AsyncMock(side_effect=RingError("status code 404"))
    garden.async_recording_url = AsyncMock(return_value=None)
    client = _client_with_devices([garden])

    with pytest.raises(RingRecordingUnavailableError):
        await client.async_describe_event("garden", "2024-01-01", "latest")


@pytest.mark.asyncio
async def test_describe_event_reraises_non_404_download_errors() -> None:
    """Non-404 download errors propagate as-is rather than attempting the share-url fallback."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(2, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    garden.async_recording_download = AsyncMock(side_effect=RingError("status code 500"))
    client = _client_with_devices([garden])

    with pytest.raises(RingRecordingUnavailableError):
        await client.async_describe_event("garden", "2024-01-01", "latest")

    garden.async_recording_url.assert_not_called()


@pytest.mark.asyncio
async def test_describe_event_raises_when_no_events_for_day() -> None:
    """A day with no matching events raises before attempting any download."""
    garden = _fake_device("Garden", history=[])
    client = _client_with_devices([garden])

    with pytest.raises(RingNoEventsFoundError):
        await client.async_describe_event("garden", "2024-01-01", "latest")

    garden.async_recording_download.assert_not_called()


@pytest.mark.asyncio
async def test_describe_event_raises_without_subscription() -> None:
    """A device with no Ring Protect subscription raises a clear, actionable error."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(1, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
        has_subscription=False,
    )
    client = _client_with_devices([garden])

    with pytest.raises(RingRecordingUnavailableError, match="Ring Protect"):
        await client.async_describe_event("garden", "2024-01-01", "latest")

    garden.async_recording_download.assert_not_called()


@pytest.mark.asyncio
async def test_describe_event_raises_when_ffmpeg_missing() -> None:
    """A missing ffmpeg/ffprobe installation surfaces as a recording-unavailable error."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(1, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    client = _client_with_devices([garden])

    with patch(
        "reachy_mini_conversation_app.ring_client.async_extract_evenly_spaced_frames",
        AsyncMock(side_effect=ring_client_module.FfmpegNotAvailableError("ffmpeg not installed")),
    ):
        with pytest.raises(RingRecordingUnavailableError, match="ffmpeg"):
            await client.async_describe_event("garden", "2024-01-01", "latest")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "expected_id"),
    [
        ("latest", 3),
        ("most recent", 3),
        ("", 3),
        ("first", 1),
        ("earliest", 1),
        ("oldest", 1),
        ("1st", 1),
        ("second", 2),
        ("2nd", 2),
        ("third", 3),
        ("siste", 3),
        ("første", 1),
    ],
)
async def test_describe_event_selects_by_latest_or_ordinal(selector: str, expected_id: int) -> None:
    """Latest/ordinal selectors (English and Norwegian) pick the right event by chronological position."""
    garden = _fake_device(
        "Garden",
        history=[
            _history_entry(3, "motion", datetime(2024, 1, 1, 16, tzinfo=timezone.utc)),
            _history_entry(2, "ding", datetime(2024, 1, 1, 12, tzinfo=timezone.utc)),
            _history_entry(1, "motion", datetime(2024, 1, 1, 8, tzinfo=timezone.utc)),
        ],
    )
    client = _client_with_devices([garden])

    with patch(
        "reachy_mini_conversation_app.ring_client.async_extract_evenly_spaced_frames",
        AsyncMock(return_value=[b"frame"]),
    ):
        event, _ = await client.async_describe_event("garden", "2024-01-01", selector)

    assert event.event_id == expected_id


@pytest.mark.asyncio
async def test_describe_event_selects_by_closest_clock_time() -> None:
    """A clock-time selector picks the event closest to that time of day."""
    garden = _fake_device(
        "Garden",
        history=[
            _history_entry(2, "motion", datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc)),
            _history_entry(1, "ding", datetime(2024, 1, 1, 9, 5, tzinfo=timezone.utc)),
        ],
    )
    client = _client_with_devices([garden])

    with patch(
        "reachy_mini_conversation_app.ring_client.async_extract_evenly_spaced_frames",
        AsyncMock(return_value=[b"frame"]),
    ):
        event, _ = await client.async_describe_event("garden", "2024-01-01", "9am")

    assert event.event_id == 1


@pytest.mark.asyncio
async def test_describe_event_raises_for_out_of_range_ordinal() -> None:
    """An ordinal beyond the number of events that day raises a clear error."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(1, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    client = _client_with_devices([garden])

    with pytest.raises(RingEventNotFoundError, match="only 1 event"):
        await client.async_describe_event("garden", "2024-01-01", "second")


@pytest.mark.asyncio
async def test_describe_event_raises_for_unrecognized_selector() -> None:
    """A selector that isn't a name, ordinal, or clock time raises a clear error."""
    garden = _fake_device(
        "Garden",
        history=[_history_entry(1, "motion", datetime(2024, 1, 1, 10, tzinfo=timezone.utc))],
    )
    client = _client_with_devices([garden])

    with pytest.raises(RingEventNotFoundError, match="banana"):
        await client.async_describe_event("garden", "2024-01-01", "banana")
