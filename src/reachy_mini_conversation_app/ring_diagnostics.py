"""Standalone connectivity check for the Ring camera tool.

Exercises the exact same RingClient code path the check_ring_camera tool uses,
so a successful run here is a strong guarantee the tool will work in the app.
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path

from reachy_mini_conversation_app.ring_client import RingClient, RingNotConfiguredError


logger = logging.getLogger(__name__)


async def async_run_ring_diagnostics(instance_path: str | None = None, *, save_dir: str | None = None) -> None:
    """Log in with the cached token, list devices, and fetch one snapshot from each."""
    client = RingClient(instance_path=instance_path)

    try:
        try:
            locations = await client.async_list_locations()
        except RingNotConfiguredError as e:
            print(f"FAILED: {e}")
            return
        except Exception as e:
            print(f"FAILED to reach Ring: {type(e).__name__}: {e}")
            return

        if not locations:
            print("Connected to Ring, but no doorbell/camera devices were found on this account.")
            return

        print(f"Connected to Ring. Found {len(locations)} device(s): {', '.join(locations)}")

        output_dir = Path(save_dir).expanduser() if save_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        for location in locations:
            try:
                jpeg_bytes = await client.async_get_device_snapshot(location)
            except Exception as e:
                print(f"  [{location}] FAILED: {type(e).__name__}: {e}")
                continue

            safe_name = "".join(c if c.isalnum() else "_" for c in location.lower())
            snapshot_path = output_dir / f"ring_snapshot_{safe_name}.jpg"
            snapshot_path.write_bytes(jpeg_bytes)
            print(f"  [{location}] OK: {len(jpeg_bytes)} bytes -> {snapshot_path}")
    finally:
        await client.async_close()


def run_ring_diagnostics(instance_path: str | None = None, *, save_dir: str | None = None) -> None:
    """Run the Ring connectivity check synchronously, for the `--ring-check` CLI flag."""
    asyncio.run(async_run_ring_diagnostics(instance_path, save_dir=save_dir))
