import time
import base64
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


class _FakeEvent:
    """A minimal realtime event: a `type` plus arbitrary attributes."""

    def __init__(self, event_type: str, **fields: Any) -> None:
        """Store the event type and any extra attributes."""
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(
    *,
    events: tuple[_FakeEvent, ...] = (),
    captured_update: dict[str, Any] | None = None,
    captured_connect: dict[str, Any] | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

        async def clear(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            if captured_connect is not None:
                captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def _fake_openai_client(captured_kwargs: dict[str, Any]) -> type:
    """Return a fake AsyncOpenAI class that records its constructor kwargs."""

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    return FakeClient


def _fake_allocator(connect_url: str, posts: list[tuple[str, dict[str, str] | None]]) -> type:
    """Return a fake httpx.AsyncClient whose POST records (url, headers) and returns `connect_url`."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"session_id": "session-123", "connect_url": connect_url}

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
            posts.append((url, headers))
            return FakeResponse()

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
            _FakeEvent(
                "conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey, how are you?"
            ),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_tool_calls_trigger_single_response(monkeypatch: Any) -> None:
    """Parallel tool calls in one turn should yield one response, not one per completed tool."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._in_flight_tool_calls = {"call_a", "call_b"}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__parallel_probe",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("call_a"))
    assert create.await_count == 0

    await handler._handle_tool_result(_completed("call_b"))
    assert create.await_count == 1


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """Unsupported profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_update=captured_update)

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_huggingface_session_uses_configured_transcription_language(monkeypatch: Any) -> None:
    """Hugging Face realtime sessions should forward the configured transcription language."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "zh")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    session = handler._get_session_config([])

    assert session["audio"]["input"]["transcription"]["language"] == "zh"


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.asyncio
async def test_build_realtime_client_uses_direct_hf_ws_url(monkeypatch: Any) -> None:
    """Hugging Face direct websocket mode should bypass the session allocator."""
    client_kwargs: dict[str, Any] = {}

    def _no_allocator(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _no_allocator)
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_TOKEN", None)
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert client_kwargs["api_key"] == "DUMMY"
    assert client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.parametrize(
    ("hf_token", "expected_header", "expected_api_key"),
    [
        ("hf-secret", {"Authorization": "Bearer hf-secret"}, "hf-secret"),
        (None, None, "DUMMY"),
    ],
)
@pytest.mark.asyncio
async def test_build_realtime_client_deployed_allocates_with_hf_token_only(
    monkeypatch: Any,
    hf_token: str | None,
    expected_header: dict[str, str] | None,
    expected_api_key: str,
) -> None:
    """Deployed mode allocates via the session URL, authenticating with HF_TOKEN only (never an OpenAI key)."""
    client_kwargs: dict[str, Any] = {}
    posts: list[tuple[str, dict[str, str] | None]] = []
    connect_url = "wss://hf.example.test/v1/realtime?session_token=allocated"
    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _fake_allocator(connect_url, posts))
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    # A stale local URL must be ignored in deployed mode.
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert posts == [("https://lb.example.test/session", expected_header)]
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    monkeypatch.setattr(handler, "_restart_session", AsyncMock(return_value=None))

    result = await handler.apply_personality("mars_rover")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_change_voice_updates_live_hf_session_without_restart(monkeypatch: Any) -> None:
    """Changing Hugging Face voice should update the active session in place."""
    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    restart = AsyncMock(return_value=None)
    monkeypatch.setattr(handler, "_restart_session", restart)

    result = await handler.change_voice("Serena")

    assert result == "Voice changed to Serena."
    assert handler.get_current_voice() == "Serena"
    restart.assert_not_awaited()
    session = captured_update["session"]
    assert session["audio"]["output"]["voice"] == "Serena"


class _StubWakeWordGate:
    """A minimal wake-word gate stand-in with a scripted active/process transition."""

    def __init__(self, *, active: bool, next_active: bool | None = None) -> None:
        self.active = active
        self._next_active = active if next_active is None else next_active

    def process(self, _sample_rate: int, _audio_frame: Any) -> None:
        self.active = self._next_active


@pytest.mark.asyncio
async def test_receive_drops_audio_while_wake_word_gate_is_inactive() -> None:
    """receive() must not forward mic audio to the backend while the wake-word gate is closed."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.connection.input_audio_buffer = AsyncMock()
    handler._wake_word_gate = _StubWakeWordGate(active=False)

    await handler.receive((16000, np.zeros(320, dtype=np.int16)))

    handler.connection.input_audio_buffer.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_forwards_audio_once_wake_word_gate_is_active() -> None:
    """receive() forwards mic audio to the backend once the wake-word gate is open."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.connection.input_audio_buffer = AsyncMock()
    handler._wake_word_gate = _StubWakeWordGate(active=True)

    await handler.receive((16000, np.zeros(320, dtype=np.int16)))

    handler.connection.input_audio_buffer.append.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_clears_input_buffer_when_gate_closes_mid_utterance() -> None:
    """Deactivating the gate mid-frame clears any trailing wake-word audio from the buffer."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.connection.input_audio_buffer = AsyncMock()
    handler._wake_word_gate = _StubWakeWordGate(active=True, next_active=False)

    await handler.receive((16000, np.zeros(320, dtype=np.int16)))

    handler.connection.input_audio_buffer.clear.assert_awaited_once()
    handler.connection.input_audio_buffer.append.assert_not_awaited()


def test_sanitize_tool_result_strips_b64_im_from_images_list() -> None:
    """A multi-image tool result (e.g. check_ring_camera) has its base64 stripped, per image."""
    tool_result = {
        "images": [
            {"label": "Garden", "b64_im": "aaaa"},
            {"label": "bod", "error": "camera offline"},
        ],
    }

    sanitized = HuggingFaceRealtimeHandler._sanitize_tool_result_for_model("check_ring_camera", tool_result)

    assert sanitized["images"][0] == {"label": "Garden", "image_attached": True}
    assert sanitized["images"][1] == {"label": "bod", "error": "camera offline"}


@pytest.mark.asyncio
async def test_inject_tool_images_sends_one_input_image_per_entry() -> None:
    """Each successfully-snapshotted image is injected as its own input_image message."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()

    await handler._inject_tool_images(
        [
            {"label": "Garden", "b64_im": "aaaa"},
            {"label": "bod", "error": "camera offline"},
        ],
    )

    assert handler.connection.conversation.item.create.await_count == 1
    sent_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert sent_item["content"][0]["image_url"] == "data:image/jpeg;base64,aaaa"


class _FakeDoorCallSession:
    """A minimal RingCallSession stand-in with an immediately-exhausted inbound stream."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.sent_audio: list[np.ndarray] = []

    async def start(self) -> None:
        self.started = True

    def send_audio(self, pcm: np.ndarray) -> None:
        self.sent_audio.append(pcm)

    async def receive_audio(self):  # noqa: ANN201 - mirrors RingCallSession's async generator signature
        return
        yield  # pragma: no cover - makes this an async generator with no items

    async def close(self) -> None:
        self.closed = True


def _door_call_deps(ring_client: Any) -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), ring_client=ring_client)


@pytest.mark.asyncio
async def test_start_door_call_returns_error_when_ring_not_configured() -> None:
    """Opening a doorbell call without a configured ring_client returns an error."""
    handler = HuggingFaceRealtimeHandler(_door_call_deps(None))

    result = await handler.start_door_call("Front Door")

    assert "error" in result


@pytest.mark.asyncio
async def test_is_door_call_active_reflects_call_lifecycle(monkeypatch: Any) -> None:
    """is_door_call_active() is False when idle, True during a call, and False again after hangup."""
    monkeypatch.setattr(hf_mod, "RingCallSession", lambda device: _FakeDoorCallSession())

    device = MagicMock()
    device.name = "Front Door"
    ring_client = MagicMock()
    ring_client.async_get_call_device = AsyncMock(return_value=device)

    handler = HuggingFaceRealtimeHandler(_door_call_deps(ring_client))
    handler.connection = AsyncMock()

    assert handler.is_door_call_active() is False

    await handler.start_door_call("Front Door")
    assert handler.is_door_call_active() is True

    await handler.end_door_call()
    assert handler.is_door_call_active() is False


@pytest.mark.asyncio
async def test_start_door_call_opens_a_session_and_nudges_the_model(monkeypatch: Any) -> None:
    """A resolved device opens a RingCallSession and queues a text-only nudge about the call."""
    fake_session = _FakeDoorCallSession()
    monkeypatch.setattr(hf_mod, "RingCallSession", lambda device: fake_session)

    device = MagicMock()
    device.name = "Front Door"
    ring_client = MagicMock()
    ring_client.async_get_call_device = AsyncMock(return_value=device)

    handler = HuggingFaceRealtimeHandler(_door_call_deps(ring_client))
    handler.connection = AsyncMock()

    result = await handler.start_door_call("Front Door")

    assert result == {"status": "connected", "location": "Front Door"}
    assert fake_session.started
    assert handler._door_call is fake_session
    handler.connection.conversation.item.create.assert_awaited_once()

    await handler.end_door_call()


@pytest.mark.asyncio
async def test_start_door_call_rejects_a_second_call_while_one_is_active(monkeypatch: Any) -> None:
    """Starting a call while already on one is rejected instead of dropping the first call."""
    monkeypatch.setattr(hf_mod, "RingCallSession", lambda device: _FakeDoorCallSession())

    device = MagicMock()
    device.name = "Front Door"
    ring_client = MagicMock()
    ring_client.async_get_call_device = AsyncMock(return_value=device)

    handler = HuggingFaceRealtimeHandler(_door_call_deps(ring_client))
    handler.connection = AsyncMock()
    await handler.start_door_call("Front Door")

    result = await handler.start_door_call("Garden")

    assert "error" in result

    await handler.end_door_call()


@pytest.mark.asyncio
async def test_end_door_call_returns_error_when_no_active_call() -> None:
    """Hanging up with no active call returns an error instead of raising."""
    handler = HuggingFaceRealtimeHandler(_door_call_deps(None))

    result = await handler.end_door_call()

    assert "error" in result


@pytest.mark.asyncio
async def test_end_door_call_closes_the_session_and_clears_state(monkeypatch: Any) -> None:
    """Hanging up closes the RingCallSession and clears the handler's call state."""
    fake_session = _FakeDoorCallSession()
    monkeypatch.setattr(hf_mod, "RingCallSession", lambda device: fake_session)

    device = MagicMock()
    device.name = "Front Door"
    ring_client = MagicMock()
    ring_client.async_get_call_device = AsyncMock(return_value=device)

    handler = HuggingFaceRealtimeHandler(_door_call_deps(ring_client))
    handler.connection = AsyncMock()
    await handler.start_door_call("Front Door")

    result = await handler.end_door_call()

    assert result == {"status": "ended", "location": "Front Door"}
    assert fake_session.closed
    assert handler._door_call is None


@pytest.mark.asyncio
async def test_assistant_audio_delta_is_forwarded_to_an_active_door_call(monkeypatch: Any) -> None:
    """The realtime event loop pushes assistant audio deltas into an active door call's outbound track."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    pcm_bytes = np.zeros(320, dtype=np.int16).tobytes()
    delta_b64 = base64.b64encode(pcm_bytes).decode("utf-8")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(events=(_FakeEvent("response.output_audio.delta", delta=delta_b64),))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    fake_session = _FakeDoorCallSession()
    handler._door_call = fake_session

    await handler._run_realtime_session()

    assert len(fake_session.sent_audio) == 1
