from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.personality import DEFAULT_OPTION
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.switch_profile import SwitchProfile


def test_switch_profile_lists_available_profiles_in_schema() -> None:
    """The tool's enum should include the built-in default option and every profile."""
    tool = SwitchProfile()

    assert DEFAULT_OPTION in tool.parameters_schema["properties"]["profile"]["enum"]
    assert "night_story_reader" in tool.parameters_schema["properties"]["profile"]["enum"]
    assert tool.needs_response is False


@pytest.mark.asyncio
async def test_switch_profile_rejects_unknown_profile() -> None:
    """Unknown profile names should fail without touching apply_personality."""
    apply_personality = AsyncMock(return_value="ok")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        apply_personality=apply_personality,
    )

    result = await SwitchProfile()(deps, profile="not_a_real_profile")

    assert result == {"error": "Unknown profile: not_a_real_profile"}
    apply_personality.assert_not_called()


@pytest.mark.asyncio
async def test_switch_profile_calls_apply_personality_with_selected_profile() -> None:
    """A known profile name should be forwarded to the apply_personality callback."""
    apply_personality = AsyncMock(return_value="Applied personality and restarted realtime session.")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        apply_personality=apply_personality,
    )

    result = await SwitchProfile()(deps, profile="night_story_reader")

    apply_personality.assert_awaited_once_with("night_story_reader")
    assert result == {
        "status": "Applied personality and restarted realtime session.",
        "profile": "night_story_reader",
    }


@pytest.mark.asyncio
async def test_switch_profile_maps_default_option_to_none() -> None:
    """Switching back to the built-in default should pass None, not the UI label."""
    apply_personality = AsyncMock(return_value="ok")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        apply_personality=apply_personality,
    )

    await SwitchProfile()(deps, profile=DEFAULT_OPTION)

    apply_personality.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_switch_profile_returns_unavailable_without_runtime_callback() -> None:
    """The tool should fail gracefully if the runtime did not inject the callback."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await SwitchProfile()(deps, profile="night_story_reader")

    assert result == {"error": "Profile switching is not available right now"}
