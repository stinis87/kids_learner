import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class TalkToDoor(Tool):
    """Open a live two-way audio call with a Ring doorbell/camera."""

    name = "talk_to_door"
    description = (
        "Open a live two-way audio call with a Ring doorbell or camera, so what you say from "
        "now on is also spoken out loud through that device's speaker, and you can hear "
        "whoever is near it. Use this when the user asks to talk to the front door, the "
        "garden, or wants to speak to someone through a specific Ring device (e.g. 'let me "
        "talk to the front door', 'tell the person outside I'm coming'). Call end_door_call "
        "once the conversation with them is over. Works no matter what language the user "
        "asks in — e.g. Norwegian 'hage' means garden and 'framsiden' means front door."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": ("The Ring device's name as configured in the Ring app (e.g. 'Front Door', 'Garden')."),
            },
        },
        "required": ["location"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Start a live doorbell call with the named Ring device."""
        location = (kwargs.get("location") or "").strip()
        if not location:
            return {"error": "location must be a non-empty string"}

        logger.info("Tool call: talk_to_door location=%s", location)

        if deps.start_door_call is None:
            return {"error": "Doorbell calls are not available in this runtime"}

        return await deps.start_door_call(location)
