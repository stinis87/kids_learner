import base64
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.ring_client import RingNotConfiguredError, RingDeviceNotFoundError
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RingCamera(Tool):
    """Check a Ring camera to see what is happening at that location right now."""

    name = "check_ring_camera"
    description = (
        "Take a fresh snapshot from one of the Ring cameras to report what is happening "
        "there right now. Use this when the user asks what is going on in the garden, "
        "at the front door, in the shed (bod), or wants to check all cameras at once. "
        "Pass the location the user asked about, or 'all' to check every camera. "
        "Works no matter what language the user asks in — e.g. Norwegian 'hage' means "
        "garden and 'framsiden' means front door."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "The Ring device's name as configured in the Ring app (e.g. 'Garden', "
                    "'Front Door', 'bod'), or 'all' to check every camera."
                ),
            },
        },
        "required": ["location"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch a snapshot from one or all Ring cameras and return it for visual analysis."""
        location = (kwargs.get("location") or "").strip()
        if not location:
            return {"error": "location must be a non-empty string"}

        logger.info("Tool call: check_ring_camera location=%s", location)

        if deps.ring_client is None:
            logger.error("Ring camera tool is not configured")
            return {"error": "Ring cameras are not configured"}

        try:
            if location.casefold() == "all":
                locations = await deps.ring_client.async_list_locations()
            else:
                locations = [location]
        except RingNotConfiguredError as e:
            logger.error("Ring is not configured: %s", e)
            return {"error": str(e)}
        except Exception as e:
            logger.error("Failed to list Ring devices: %s", e)
            return {"error": f"Failed to list Ring devices: {type(e).__name__}: {e}"}

        snapshots = await asyncio.gather(
            *(self._snapshot(deps, camera_location) for camera_location in locations),
        )
        return {"images": list(snapshots)}

    async def _snapshot(self, deps: ToolDependencies, location: str) -> Dict[str, Any]:
        """Fetch one camera's snapshot, turning any failure into a per-camera error entry."""
        assert deps.ring_client is not None
        try:
            jpeg_bytes = await deps.ring_client.async_get_device_snapshot(location)
        except (RingNotConfiguredError, RingDeviceNotFoundError) as e:
            logger.warning("Ring snapshot failed for %s: %s", location, e)
            return {"label": location, "error": str(e)}
        except Exception as e:
            logger.error("Ring snapshot failed for %s: %s", location, e)
            return {"label": location, "error": f"{type(e).__name__}: {e}"}

        return {"label": location, "b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
