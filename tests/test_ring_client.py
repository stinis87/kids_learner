"""Tests for the Ring client's location alias resolution and device lookup."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from ring_doorbell import RingError

from reachy_mini_conversation_app import ring_client as ring_client_module
from reachy_mini_conversation_app.ring_client import RingClient, RingDeviceNotFoundError


def _fake_device(name: str, device_id: int = 123, history: list[dict[str, object]] | None = None) -> MagicMock:
    device = MagicMock()
    device.name = name
    device.id = device_id
    device.async_history = AsyncMock(return_value=history or [])
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
    assert events["Front Door"].event_id == 3
