"""WebRTC audio bridge for two-way audio calls with a Ring doorbell/camera.

Wraps ``aiortc``'s WebRTC media handling with ``ring_doorbell``'s live-view
signaling (``RingDoorBell.generate_webrtc_stream``) to open a two-way audio
session with a Ring device — the same mechanism the Ring app itself uses for
live view and two-way talk. Bridges mono PCM16 audio in both directions at
``CALL_SAMPLE_RATE`` so it can be wired into the realtime conversation loop's
existing mic-in/speaker-out pipeline.
"""

from __future__ import annotations
import time
import asyncio
import logging
import fractions
import contextlib
from collections.abc import AsyncIterator

import numpy as np
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription
from numpy.typing import NDArray
from ring_doorbell import RingError
from av.audio.resampler import AudioResampler
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from ring_doorbell.doorbot import RingDoorBell


logger = logging.getLogger(__name__)

# The realtime conversation loop bridges audio at this rate in both
# directions (matches HuggingFaceRealtimeHandler.SAMPLE_RATE), so no
# caller-side resampling is needed.
CALL_SAMPLE_RATE = 16000

_OUTBOUND_FRAME_MS = 20
_OUTBOUND_FRAME_SAMPLES = CALL_SAMPLE_RATE * _OUTBOUND_FRAME_MS // 1000


class RingCallError(Exception):
    """Raised when a Ring WebRTC call session fails to start or breaks down."""


class _PushAudioTrack(MediaStreamTrack):
    """Outbound audio track fed by pushing mono PCM16 frames at `CALL_SAMPLE_RATE`.

    Emits fixed-size 20ms frames on a steady clock, as real-time WebRTC audio
    requires, padding with silence when nothing has been pushed yet.
    """

    kind = "audio"

    def __init__(self) -> None:
        """Start with an empty outbound buffer; pacing begins on the first `recv`."""
        super().__init__()
        self._buffer: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._pts = 0
        self._next_frame_time: float | None = None

    def push(self, pcm: NDArray[np.int16]) -> None:
        """Append mono PCM16 samples at `CALL_SAMPLE_RATE` to the outbound buffer."""
        self._buffer = np.concatenate([self._buffer, pcm.astype(np.int16, copy=False)])

    async def recv(self) -> AudioFrame:
        """Return the next 20ms outbound frame, generating silence when the buffer is empty."""
        now = time.monotonic()
        if self._next_frame_time is None:
            self._next_frame_time = now
        else:
            self._next_frame_time += _OUTBOUND_FRAME_MS / 1000
            wait = self._next_frame_time - now
            if wait > 0:
                await asyncio.sleep(wait)

        if len(self._buffer) >= _OUTBOUND_FRAME_SAMPLES:
            chunk, self._buffer = (
                self._buffer[:_OUTBOUND_FRAME_SAMPLES],
                self._buffer[_OUTBOUND_FRAME_SAMPLES:],
            )
        else:
            chunk = np.zeros(_OUTBOUND_FRAME_SAMPLES, dtype=np.int16)

        frame = AudioFrame(format="s16", layout="mono", samples=_OUTBOUND_FRAME_SAMPLES)
        frame.sample_rate = CALL_SAMPLE_RATE
        frame.planes[0].update(chunk.tobytes())
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, CALL_SAMPLE_RATE)
        self._pts += _OUTBOUND_FRAME_SAMPLES
        return frame


class RingCallSession:
    """A live two-way audio call with a Ring device, established over WebRTC."""

    def __init__(self, device: RingDoorBell) -> None:
        """Set up the peer connection and outbound track; nothing is sent until `start`."""
        self._device = device
        self._pc = RTCPeerConnection()
        self._outbound_track = _PushAudioTrack()
        self._pc.addTrack(self._outbound_track)
        self._resampler = AudioResampler(format="s16", layout="mono", rate=CALL_SAMPLE_RATE)
        self._inbound_frames: asyncio.Queue[NDArray[np.int16] | None] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

        @self._pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio" and self._reader_task is None:
                self._reader_task = asyncio.create_task(self._read_inbound(track))

    async def start(self) -> None:
        """Open the WebRTC session via the device's live-view signaling."""
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        assert self._pc.localDescription is not None

        try:
            answer_sdp = await self._device.generate_webrtc_stream(self._pc.localDescription.sdp)
        except RingError as e:
            await self._pc.close()
            raise RingCallError(f"Ring rejected the call for '{self._device.name}': {e}") from e

        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

    def send_audio(self, pcm: NDArray[np.int16]) -> None:
        """Queue mono PCM16 audio at `CALL_SAMPLE_RATE` to play out the Ring device's speaker."""
        if self._closed:
            return
        self._outbound_track.push(pcm)

    async def receive_audio(self) -> AsyncIterator[NDArray[np.int16]]:
        """Yield mono PCM16 frames at `CALL_SAMPLE_RATE` captured from the Ring device's mic."""
        while True:
            frame = await self._inbound_frames.get()
            if frame is None:
                return
            yield frame

    async def _read_inbound(self, track: MediaStreamTrack) -> None:
        """Pull frames from the remote audio track, resample them, and queue PCM16 chunks."""
        try:
            while True:
                frame = await track.recv()
                if not isinstance(frame, AudioFrame):
                    continue
                for resampled in self._resampler.resample(frame):
                    pcm = np.frombuffer(bytes(resampled.planes[0]), dtype=np.int16, count=resampled.samples)
                    await self._inbound_frames.put(pcm.copy())
        except MediaStreamError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Ring call inbound audio reader stopped for '%s': %s", self._device.name, e)
        finally:
            await self._inbound_frames.put(None)

    async def close(self) -> None:
        """Close the WebRTC session and stop the inbound reader."""
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        await self._pc.close()
        await self._inbound_frames.put(None)
