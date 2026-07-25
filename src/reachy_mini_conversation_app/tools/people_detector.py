"""Local, zero-network-per-call people/pose detection using MediaPipe Pose Landmarker."""

import os
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
import mediapipe as mp


logger = logging.getLogger(__name__)

# Landmark indices for the upper body (shoulders, elbows, wrists, hips), matching
# MediaPipe's pose landmark topology: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _model_cache_path() -> Path:
    """Return where the pose landmarker model is cached, downloading it on first use."""
    cache_home = os.getenv("XDG_CACHE_HOME")
    cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return cache_root / "reachy_mini_conversation_app" / _MODEL_FILENAME


def ensure_pose_model_downloaded() -> str:
    """Download the pose landmarker model to the local cache if it isn't there yet, and return its path."""
    model_path = _model_cache_path()
    if model_path.exists():
        return str(model_path)

    logger.info("Downloading pose landmarker model to %s", model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    with httpx.stream("GET", _MODEL_URL, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    tmp_path.rename(model_path)
    return str(model_path)


class PeopleDetector:
    """Detects each visible person's upper-body pose landmarks in a camera frame."""

    def __init__(self, model_path: str, max_people: int = 5):
        """Load the pose landmarker model, ready to detect up to max_people at once."""
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=max_people,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[list[tuple[int, int]]]:
        """Return each detected person's upper-body landmark pixel coordinates."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp_image)

        height, width = frame.shape[:2]
        people: list[list[tuple[int, int]]] = []
        for person_landmarks in result.pose_landmarks:
            people.append(
                [
                    (int(person_landmarks[idx].x * width), int(person_landmarks[idx].y * height))
                    for idx in UPPER_BODY_LANDMARKS
                ]
            )
        return people
