"""Tests for the standalone Ring connectivity diagnostic (--ring-check)."""

from unittest.mock import AsyncMock, patch

import pytest

from reachy_mini_conversation_app.ring_client import RingNotConfiguredError
from reachy_mini_conversation_app.ring_diagnostics import async_run_ring_diagnostics


@pytest.mark.asyncio
async def test_reports_missing_login_without_crashing(capsys: pytest.CaptureFixture[str]) -> None:
    """No cached token yet -> a clear message, not an exception."""
    with patch(
        "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_list_locations",
        AsyncMock(side_effect=RingNotConfiguredError("No cached Ring login found.")),
    ):
        await async_run_ring_diagnostics()

    assert "FAILED" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_saves_one_jpeg_per_device(tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    """A successful run lists devices and writes one snapshot file per device."""
    with (
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_list_locations",
            AsyncMock(return_value=["Garden", "Front Door"]),
        ),
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_get_device_snapshot",
            AsyncMock(return_value=b"\xff\xd8jpeg\xff\xd9"),
        ),
    ):
        await async_run_ring_diagnostics(save_dir=str(tmp_path))

    saved_files = sorted(p.name for p in tmp_path.iterdir())
    assert saved_files == ["ring_snapshot_front_door.jpg", "ring_snapshot_garden.jpg"]
    assert "OK" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_default_save_dir_is_ring_images_under_cwd(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit save_dir, snapshots land in a gitignored ring_images/ folder."""
    monkeypatch.chdir(tmp_path)
    with (
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_list_locations",
            AsyncMock(return_value=["Garden"]),
        ),
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_get_device_snapshot",
            AsyncMock(return_value=b"\xff\xd8jpeg\xff\xd9"),
        ),
    ):
        await async_run_ring_diagnostics()

    saved_files = sorted(p.name for p in (tmp_path / "ring_images").iterdir())  # type: ignore[operator]
    assert saved_files == ["ring_snapshot_garden.jpg"]


@pytest.mark.asyncio
async def test_one_failing_device_does_not_stop_the_others(tmp_path: object) -> None:
    """A single device erroring out still lets the diagnostic finish and save the rest."""

    async def snapshot(location: str) -> bytes:
        if location == "bod":
            raise RuntimeError("camera offline")
        return b"\xff\xd8jpeg\xff\xd9"

    with (
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_list_locations",
            AsyncMock(return_value=["Garden", "bod"]),
        ),
        patch(
            "reachy_mini_conversation_app.ring_diagnostics.RingClient.async_get_device_snapshot",
            AsyncMock(side_effect=snapshot),
        ),
    ):
        await async_run_ring_diagnostics(save_dir=str(tmp_path))

    saved_files = sorted(p.name for p in tmp_path.iterdir())
    assert saved_files == ["ring_snapshot_garden.jpg"]
