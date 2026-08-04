import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class EndDoorCall(Tool):
    """End the active live doorbell call, if any."""

    name = "end_door_call"
    description = (
        "End the current live two-way audio call opened by talk_to_door or an autonomous "
        "doorbell answer, so audio stops being routed to and from the Ring device. Use this "
        "once the conversation with the person at the door is over — e.g. they said goodbye, "
        "or the user asks to hang up."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """End the active doorbell call."""
        logger.info("Tool call: end_door_call")

        if deps.end_door_call is None:
            return {"error": "Doorbell calls are not available in this runtime"}

        return await deps.end_door_call()
