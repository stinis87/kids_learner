"""Tests for the Ring client's location alias resolution and device lookup."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.ring_client import RingClient, RingDeviceNotFoundError


def _fake_device(name: str, jpeg_bytes: bytes = b"\xff\xd8jpeg\xff\xd9") -> MagicMock:
    device = MagicMock()
    device.name = name
    device.async_get_snapshot = AsyncMock(return_value=jpeg_bytes)
    return device


def _client_with_devices(devices: list[MagicMock]) -> RingClient:
    client = RingClient()
    ring = MagicMock()
    ring.async_update_devices = AsyncMock()
    ring.video_devices = MagicMock(return_value=devices)
    client._get_ring = AsyncMock(return_value=ring)
    return client


@pytest.mark.asyncio
async def test_exact_device_name_matches_case_insensitively() -> None:
    """A location matching a device name directly (any case) resolves to that device."""
    client = _client_with_devices([_fake_device("Garden"), _fake_device("Front Door")])

    await client.async_get_device_snapshot("garden")

    client._get_ring.return_value.video_devices()[0].async_get_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_norwegian_alias_hage_resolves_to_garden() -> None:
    """'Hage' is a Norwegian synonym for the 'Garden' device."""
    garden = _fake_device("Garden")
    client = _client_with_devices([garden, _fake_device("Front Door"), _fake_device("bod")])

    await client.async_get_device_snapshot("Hage")

    garden.async_get_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_norwegian_alias_framsiden_resolves_to_front_door() -> None:
    """'Framsiden' is a Norwegian synonym for the 'Front Door' device."""
    front_door = _fake_device("Front Door")
    client = _client_with_devices([_fake_device("Garden"), front_door, _fake_device("bod")])

    await client.async_get_device_snapshot("framsiden")

    front_door.async_get_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_location_raises_with_known_device_names() -> None:
    """An unmatched location raises, listing the actually configured device names."""
    client = _client_with_devices([_fake_device("Garden"), _fake_device("bod")])

    with pytest.raises(RingDeviceNotFoundError, match="Garden.*bod|bod.*Garden"):
        await client.async_get_device_snapshot("attic")
