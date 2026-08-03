"""Extract evenly-spaced still frames from a video clip via the `ffmpeg`/`ffprobe` binaries.

Used to describe Ring recordings: the app's vision pipeline only ever sends still
images to the model (see `huggingface_realtime._inject_tool_images`), so a clip is
never sent whole — a handful of frames spread across its duration stand in for it.
Nothing is written to disk beyond a per-call temporary directory, removed immediately
after frame extraction.
"""

from __future__ import annotations
import shutil
import asyncio
import logging
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)

FFMPEG_BINARY = "ffmpeg"
FFPROBE_BINARY = "ffprobe"

# Ring clips are typically 30-60s; used only if ffprobe can't report a duration, so
# frame sampling still spreads out across a plausible clip length rather than bunching.
_FALLBACK_DURATION_S = 30.0


class FfmpegNotAvailableError(Exception):
    """Raised when the `ffmpeg`/`ffprobe` binaries aren't installed."""


async def async_extract_evenly_spaced_frames(video_bytes: bytes, frame_count: int) -> list[bytes]:
    """Return up to `frame_count` JPEG frames sampled at even intervals across `video_bytes`."""
    if shutil.which(FFMPEG_BINARY) is None or shutil.which(FFPROBE_BINARY) is None:
        raise FfmpegNotAvailableError(
            "ffmpeg/ffprobe are not installed; required to extract frames from Ring recordings."
        )

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        video_path = tmp_dir / "clip.mp4"
        video_path.write_bytes(video_bytes)

        duration_s = await _probe_duration_seconds(video_path)
        frames: list[bytes] = []
        for index in range(frame_count):
            # Sample at each segment's midpoint rather than its edge, so the first and
            # last frames aren't the clip's often-blank very start/end.
            timestamp_s = duration_s * (index + 0.5) / frame_count
            frame_path = tmp_dir / f"frame_{index}.jpg"
            await _run_ffmpeg(video_path, timestamp_s, frame_path)
            if frame_path.exists():
                frames.append(frame_path.read_bytes())

        return frames


async def _probe_duration_seconds(video_path: Path) -> float:
    """Return the clip's duration in seconds, falling back to a typical clip length on failure."""
    process = await asyncio.create_subprocess_exec(
        FFPROBE_BINARY,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        logger.warning("ffprobe failed to read clip duration: %s", stderr.decode(errors="replace"))
        return _FALLBACK_DURATION_S
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return _FALLBACK_DURATION_S


async def _run_ffmpeg(video_path: Path, timestamp_s: float, frame_path: Path) -> None:
    """Extract a single JPEG frame at `timestamp_s` into `frame_path`."""
    process = await asyncio.create_subprocess_exec(
        FFMPEG_BINARY,
        "-ss",
        f"{timestamp_s:.2f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(frame_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        logger.warning("ffmpeg failed to extract frame at %.2fs: %s", timestamp_s, stderr.decode(errors="replace"))
