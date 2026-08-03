"""Tests for the ffmpeg-based frame sampling used to describe Ring recordings."""

from pathlib import Path
from unittest.mock import patch

import pytest

from reachy_mini_conversation_app.video_frames import _resolve_binary, _sample_timestamps


def test_resolve_binary_prefers_path_lookup() -> None:
    """A binary found on PATH is used as-is, without checking fallback directories."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        assert _resolve_binary("ffmpeg") == "/usr/bin/ffmpeg"


def test_resolve_binary_falls_back_to_common_install_dirs(tmp_path: Path) -> None:
    """When PATH lookup fails, well-known package-manager directories are checked."""
    fallback_binary = tmp_path / "ffmpeg"
    fallback_binary.write_text("#!/bin/sh\n")

    with (
        patch("shutil.which", return_value=None),
        patch("reachy_mini_conversation_app.video_frames._FALLBACK_BINARY_DIRS", (str(tmp_path),)),
    ):
        assert _resolve_binary("ffmpeg") == str(fallback_binary)


def test_resolve_binary_returns_none_when_nowhere_found(tmp_path: Path) -> None:
    """Returns None rather than raising when the binary truly isn't installed."""
    with (
        patch("shutil.which", return_value=None),
        patch("reachy_mini_conversation_app.video_frames._FALLBACK_BINARY_DIRS", (str(tmp_path),)),
    ):
        assert _resolve_binary("ffmpeg") is None


def test_sample_timestamps_returns_empty_for_zero_frames() -> None:
    """Asking for zero frames returns an empty list rather than dividing by zero."""
    assert _sample_timestamps(duration_s=30.0, frame_count=0) == []


def test_sample_timestamps_single_frame_is_at_the_midpoint() -> None:
    """A single requested frame lands exactly in the middle of the clip."""
    (timestamp_s,) = _sample_timestamps(duration_s=30.0, frame_count=1)
    assert timestamp_s == pytest.approx(15.0)


def test_sample_timestamps_avoids_the_very_start_and_end() -> None:
    """Sampled timestamps stay clear of the clip's blank pre-/post-roll edges."""
    duration_s = 30.0
    timestamps = _sample_timestamps(duration_s=duration_s, frame_count=2)

    assert timestamps[0] > 0.0
    assert timestamps[-1] < duration_s


def test_sample_timestamps_are_spread_apart_not_adjacent() -> None:
    """Consecutive samples are separated by a meaningful gap, not near-duplicates."""
    timestamps = _sample_timestamps(duration_s=30.0, frame_count=2)

    assert timestamps[1] - timestamps[0] > 5.0


def test_sample_timestamps_are_evenly_spaced_across_middle_portion() -> None:
    """With more than two frames, consecutive gaps are equal-sized."""
    timestamps = _sample_timestamps(duration_s=60.0, frame_count=4)

    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert gaps == pytest.approx([gaps[0]] * len(gaps))
