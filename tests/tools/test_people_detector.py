"""Tests for the pose landmarker model caching in people_detector."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from reachy_mini_conversation_app.tools import people_detector
from reachy_mini_conversation_app.tools.people_detector import ensure_pose_model_downloaded


def test_ensure_pose_model_downloaded_skips_download_when_cached(tmp_path: Path) -> None:
    """An already-cached model is reused without hitting the network."""
    cached_model = tmp_path / "reachy_mini_conversation_app" / "pose_landmarker_lite.task"
    cached_model.parent.mkdir(parents=True)
    cached_model.write_bytes(b"cached")

    with (
        patch.object(people_detector, "_model_cache_path", return_value=cached_model),
        patch("httpx.stream") as mock_stream,
    ):
        result = ensure_pose_model_downloaded()

    assert result == str(cached_model)
    mock_stream.assert_not_called()


def test_ensure_pose_model_downloaded_fetches_missing_model(tmp_path: Path) -> None:
    """A missing model is streamed from the model URL and written to the cache path."""
    model_path = tmp_path / "reachy_mini_conversation_app" / "pose_landmarker_lite.task"

    response = MagicMock()
    response.iter_bytes.return_value = [b"chunk-1", b"chunk-2"]
    response.raise_for_status.return_value = None
    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = response
    stream_ctx.__exit__.return_value = False

    with (
        patch.object(people_detector, "_model_cache_path", return_value=model_path),
        patch("httpx.stream", return_value=stream_ctx) as mock_stream,
    ):
        result = ensure_pose_model_downloaded()

    mock_stream.assert_called_once()
    assert result == str(model_path)
    assert model_path.read_bytes() == b"chunk-1chunk-2"
