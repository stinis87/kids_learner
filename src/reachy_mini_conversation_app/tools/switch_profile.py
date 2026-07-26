import logging
from typing import Any, Dict

from reachy_mini_conversation_app.personality import DEFAULT_OPTION, list_personalities
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SwitchProfile(Tool):
    """Switch Reachy's active personality/profile during the conversation."""

    name = "switch_profile"
    # apply_personality already restarts the session and plays the new profile's own
    # greeting, so a spoken follow-up in the old persona would just be confusing.
    needs_response = False
    parameters_schema: Dict[str, Any] = {}

    def __init__(self) -> None:
        """Build the description and enum from the profiles available right now."""
        self._available_profiles = [DEFAULT_OPTION, *list_personalities()]
        self.description = (
            "Switch Reachy's active personality/profile when the user asks to change how it behaves or wants a "
            "different kind of companion, e.g. 'let's read a bedtime story', 'switch to night story reader mode', "
            "or 'go back to normal'. Pick whichever available profile best matches what the user asked for, even "
            "if they don't say its exact name. Works no matter what language the user asks in."
        )
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": self._available_profiles,
                    "description": f"Exact profile name to switch to. Available: {', '.join(self._available_profiles)}.",
                },
            },
            "required": ["profile"],
        }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Apply the requested personality profile via the active handler."""
        profile = str(kwargs.get("profile") or "").strip()
        if not profile:
            return {"error": "profile must be a non-empty string"}
        if profile not in self._available_profiles:
            logger.warning("switch_profile: unknown profile requested: %s", profile)
            return {"error": f"Unknown profile: {profile}"}
        if deps.apply_personality is None:
            logger.error("switch_profile: no apply_personality callback configured")
            return {"error": "Profile switching is not available right now"}

        logger.info("Tool call: switch_profile profile=%s", profile)
        status = await deps.apply_personality(None if profile == DEFAULT_OPTION else profile)
        return {"status": status, "profile": profile}
