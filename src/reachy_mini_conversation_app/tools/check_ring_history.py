import base64
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.ring_client import (
    RingEventNotFoundError,
    RingNoEventsFoundError,
    RingNotConfiguredError,
    RingDeviceNotFoundError,
    RingDayNotRecognizedError,
    RingRecordingUnavailableError,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RingHistory(Tool):
    """Answer retroactive questions about a Ring camera's past motion/doorbell events."""

    name = "check_ring_history"
    description = (
        "Look up what happened at a Ring camera on a past day — e.g. 'was anyone at the "
        "front door today?', 'how many times did the doorbell ring yesterday?', or 'what "
        "happened in the garden on 2026-01-05?'. Pass 'today', 'yesterday', 'day before "
        "yesterday', or a YYYY-MM-DD date. By default this only returns event counts and "
        "times, which is fast — use this first for a broad question. If the user then "
        "asks about one specific event from that list (e.g. 'what happened at the second "
        "one?', 'describe the one at 2pm', 'what was the doorbell ring about?'), call "
        "again with describe_event set to identify it — this downloads that event's "
        "recorded clip and looks at several frames from it, so it is slower (can take a "
        "little while) and requires an active Ring Protect subscription. Before calling "
        "with describe_event set, briefly tell the user you're checking and it might take "
        "a moment — do not do this for a plain count/summary lookup, which is fast enough "
        "to answer right away. Works no matter what language the user asks in — e.g. "
        "Norwegian 'hage' means garden, 'framsiden' means front door, and 'i dag'/'i går'/"
        "'i forgårs' mean today/yesterday/day before yesterday."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "The Ring device's name as configured in the Ring app (e.g. 'Garden', 'Front Door', 'bod')."
                ),
            },
            "day": {
                "type": "string",
                "description": (
                    "'today', 'yesterday', 'day before yesterday', or a YYYY-MM-DD date "
                    "(also accepts Norwegian 'i dag', 'i går', 'i forgårs')."
                ),
            },
            "describe_event": {
                "type": "string",
                "description": (
                    "Set this to also look at frames from one specific event's recorded "
                    "clip, describing what actually happened rather than just reporting "
                    "that something happened. Accepts 'latest'/'most recent', 'earliest'/"
                    "'oldest', an ordinal like 'first'/'second'/'third' (counted "
                    "chronologically from the day's earliest event), or a clock time like "
                    "'14:00' or '2pm' to match the closest event. Slower than a plain "
                    "lookup — tell the user to expect a short wait before setting this. "
                    "Leave unset for a fast summary with no clip download."
                ),
            },
        },
        "required": ["location", "day"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Report a day's Ring events for a location, optionally describing one specific event."""
        location = (kwargs.get("location") or "").strip()
        day = (kwargs.get("day") or "").strip()
        describe_event = (kwargs.get("describe_event") or "").strip()
        if not location or not day:
            return {"error": "location and day must be non-empty strings"}

        logger.info(
            "Tool call: check_ring_history location=%s day=%s describe_event=%r",
            location,
            day,
            describe_event,
        )

        if deps.ring_client is None:
            logger.error("Ring history tool is not configured")
            return {"error": "Ring cameras are not configured"}

        try:
            summary = await deps.ring_client.async_get_history_for_day(location, day)
        except (RingNotConfiguredError, RingDeviceNotFoundError, RingDayNotRecognizedError) as e:
            logger.warning("Ring history lookup failed for %s/%s: %s", location, day, e)
            return {"error": str(e)}
        except Exception as e:
            logger.error("Ring history lookup failed for %s/%s: %s", location, day, e)
            return {"error": f"Failed to fetch Ring history: {type(e).__name__}: {e}"}

        result: Dict[str, Any] = {
            "device_name": summary.device_name,
            "day": summary.day.isoformat(),
            "event_count": len(summary.events),
            "events": [{"kind": event.kind, "created_at": event.created_at.isoformat()} for event in summary.events],
        }

        if describe_event and summary.events:
            images, description_error = await self._describe_event(deps, location, day, describe_event)
            if images is not None:
                result["images"] = images
            elif description_error is not None:
                result["description_error"] = description_error

        return result

    async def _describe_event(
        self, deps: ToolDependencies, location: str, day: str, selector: str
    ) -> tuple[list[Dict[str, Any]] | None, str | None]:
        """Fetch frames from the event matching `selector`.

        Returns `(images, None)` on success, or `(None, error_message)` if the clip
        couldn't be described — surfaced to the model rather than swallowed, so it
        can explain to the user (e.g. "no Ring Protect subscription") instead of
        just reporting the event happened with no further detail.
        """
        assert deps.ring_client is not None
        try:
            event, frames = await deps.ring_client.async_describe_event(location, day, selector)
        except (RingNoEventsFoundError, RingEventNotFoundError, RingRecordingUnavailableError) as e:
            logger.warning("Could not describe Ring event '%s' for %s/%s: %s", selector, location, day, e)
            return None, str(e)
        except Exception as e:
            logger.error("Could not describe Ring event '%s' for %s/%s: %s", selector, location, day, e)
            return None, f"Failed to describe the event: {type(e).__name__}: {e}"

        images = [
            {
                "label": f"{event.kind} at {event.created_at.isoformat()}, frame {index + 1}/{len(frames)}",
                "b64_im": base64.b64encode(frame).decode("utf-8"),
            }
            for index, frame in enumerate(frames)
        ]
        return images, None
