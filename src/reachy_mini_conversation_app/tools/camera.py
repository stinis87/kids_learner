import base64
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live: call this tool again every single time the user asks a new visual question, "
        "even if you already took a picture earlier in the conversation — the scene may have changed "
        "(they may be holding something new, have moved, etc.), and an older picture in the conversation "
        "history is never a substitute for a fresh one. "
        "Never say you need to take a picture or ask for permission/confirmation first — there is no "
        "confirmation step, just call this tool right away in the same turn and then answer from the result. "
        "If the user asks you to look without saying at what, do not ask for clarification, call this tool and describe what you see. "
        "This applies no matter what language the user asks in — Norwegian requests like 'hva er dette', "
        "'hva holder jeg', 'hva ser du', 'ser du...', 'beskriv hva du ser', and 'hvem er jeg' are the same "
        "kind of request as their English equivalents and must always trigger a fresh call to this tool, "
        "not a spoken guess or a reused earlier picture."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to observe or ask about in the picture. "
                    "Examples: what is the user holding, describe the user's outfit, "
                    "what do you see around you, how does the user look today."
                ),
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and return the base64-encoded JPEG."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
