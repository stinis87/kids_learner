"""Tests for RingCallSession's WebRTC signaling and audio bridging."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from collections.abc import Callable

import numpy as np
import pytest
from av import AudioFrame
from ring_doorbell import RingError
from aiortc.mediastreams import MediaStreamError

from reachy_mini_conversation_app.ring_call import (
    CALL_SAMPLE_RATE,
    _OUTBOUND_FRAME_SAMPLES,
    RingCallError,
    RingCallSession,
    _PushAudioTrack,
)


class _FakePeerConnection:
    """Stand-in for aiortc's RTCPeerConnection exposing just what RingCallSession uses."""

    def __init__(self) -> None:
        self.localDescription: SimpleNamespace | None = None
        self.remoteDescription: object | None = None
        self.closed = False
        self.added_tracks: list[object] = []
        self._track_handler: Callable[[object], None] | None = None

    def on(self, event: str) -> Callable[[Callable[[object], None]], Callable[[object], None]]:
        """Register `func` as the handler for `event`, mirroring aiortc's decorator API."""

        def decorator(func: Callable[[object], None]) -> Callable[[object], None]:
            if event == "track":
                self._track_handler = func
            return func

        return decorator

    def addTrack(self, track: object) -> None:
        self.added_tracks.append(track)

    async def createOffer(self) -> SimpleNamespace:
        return SimpleNamespace(sdp=_VALID_OFFER_SDP, type="offer")

    async def setLocalDescription(self, description: SimpleNamespace) -> None:
        self.localDescription = description

    async def setRemoteDescription(self, description: object) -> None:
        self.remoteDescription = description

    async def close(self) -> None:
        self.closed = True

    def trigger_track(self, track: object) -> None:
        if self._track_handler is not None:
            self._track_handler(track)


def _fake_device(name: str = "Front Door") -> MagicMock:
    device = MagicMock()
    device.name = name
    device.generate_webrtc_stream = AsyncMock(return_value="answer-sdp")
    device.keep_alive_webrtc_stream = AsyncMock()
    return device


_VALID_OFFER_SDP = "v=0\r\no=- 46117317 2 IN IP4 127.0.0.1\r\ns=-\r\n"


class _FakeAudioTrack:
    """A remote audio track yielding preset frames, then raising MediaStreamError."""

    kind = "audio"

    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)

    async def recv(self) -> AudioFrame:
        if not self._frames:
            raise MediaStreamError
        return self._frames.pop(0)


def _stereo_frame(sample_rate: int = 48000, samples: int = 960) -> AudioFrame:
    frame = AudioFrame(format="s16", layout="stereo", samples=samples)
    frame.sample_rate = sample_rate
    plane = frame.planes[0]
    frame.planes[0].update(np.zeros(plane.buffer_size, dtype=np.uint8).tobytes())
    return frame


@pytest.mark.asyncio
async def test_start_sends_offer_and_sets_remote_answer() -> None:
    """Starting a call sends the SDP offer to the device and applies the returned answer."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        await session.start()

    device.generate_webrtc_stream.assert_awaited_once_with(_VALID_OFFER_SDP, keep_alive_timeout=30)
    assert fake_pc.remoteDescription is not None
    assert fake_pc.added_tracks  # outbound track was registered
    await session.close()


@pytest.mark.asyncio
async def test_start_extracts_session_id_and_schedules_keep_alive() -> None:
    """Starting a call parses the SDP session id and starts a keep-alive background task."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        await session.start()

        assert session._session_id == "46117317"
        assert session._keep_alive_task is not None
        assert not session._keep_alive_task.done()

        await session.close()

    assert session._keep_alive_task.cancelled()


@pytest.mark.asyncio
async def test_keep_alive_loop_pings_ring_before_the_session_times_out(monkeypatch: object) -> None:
    """The keep-alive loop periodically pings Ring so the call outlives keep_alive_timeout."""
    import reachy_mini_conversation_app.ring_call as ring_call_mod

    monkeypatch.setattr(ring_call_mod, "_KEEP_ALIVE_INTERVAL_S", 0.01)

    fake_pc = _FakePeerConnection()
    device = _fake_device()
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        await session.start()

        for _ in range(50):
            if device.keep_alive_webrtc_stream.await_count > 0:
                break
            await asyncio.sleep(0.01)

        await session.close()

    device.keep_alive_webrtc_stream.assert_awaited_with("46117317")


@pytest.mark.asyncio
async def test_start_raises_ring_call_error_and_closes_pc_on_ring_error() -> None:
    """A Ring signaling failure raises RingCallError and cleans up the peer connection."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    device.generate_webrtc_stream = AsyncMock(side_effect=RingError("device offline"))
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        with pytest.raises(RingCallError, match="device offline"):
            await session.start()

    assert fake_pc.closed


@pytest.mark.asyncio
async def test_send_audio_is_a_noop_after_close() -> None:
    """Audio pushed after close() is dropped instead of being queued."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        await session.close()
        session.send_audio(np.zeros(CALL_SAMPLE_RATE // 100, dtype=np.int16))

    assert session._outbound_track._buffer.size == 0


@pytest.mark.asyncio
async def test_receive_audio_resamples_inbound_frames_and_stops_on_close() -> None:
    """Inbound frames from the remote track are resampled to CALL_SAMPLE_RATE mono PCM16."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    frames = [_stereo_frame(), _stereo_frame()]

    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        fake_pc.trigger_track(_FakeAudioTrack(frames))

        received: list[np.ndarray] = []
        async for pcm in session.receive_audio():
            assert pcm.dtype == np.int16
            received.append(pcm)

    assert len(received) == len(frames)


@pytest.mark.asyncio
async def test_push_audio_track_emits_paced_silence_when_buffer_empty() -> None:
    """With nothing pushed yet, recv() still returns a correctly shaped silent frame."""
    track = _PushAudioTrack()
    frame = await track.recv()
    assert frame.sample_rate == CALL_SAMPLE_RATE
    assert frame.samples == _OUTBOUND_FRAME_SAMPLES
    pcm = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16, count=frame.samples)
    assert np.all(pcm == 0)


@pytest.mark.asyncio
async def test_push_audio_track_returns_pushed_samples() -> None:
    """Samples passed to push() come back out unchanged from the next recv()."""
    track = _PushAudioTrack()
    pushed = np.full(_OUTBOUND_FRAME_SAMPLES, 1234, dtype=np.int16)
    track.push(pushed)

    frame = await track.recv()
    pcm = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16, count=frame.samples)
    assert np.array_equal(pcm, pushed)


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """Calling close() twice does not raise or double-close the peer connection."""
    fake_pc = _FakePeerConnection()
    device = _fake_device()
    with patch("reachy_mini_conversation_app.ring_call.RTCPeerConnection", return_value=fake_pc):
        session = RingCallSession(device)
        await session.close()
        await session.close()

    assert fake_pc.closed
