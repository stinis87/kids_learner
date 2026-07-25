"""Local wake-word gate: mic audio stops reaching the conversation backend once put to sleep.

Uses openWakeWord's pretrained ONNX models (CPU-only, no custom training). The gate starts
open (Reachy awake and listening); saying the wake word toggles it closed, saying it again
toggles it back open.
"""

from __future__ import annotations
import time
import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80ms at 16kHz, openWakeWord's native frame size
_REFRACTORY_SECONDS = 1.5  # ignore further detections right after a toggle, to not re-trigger on the same utterance


class WakeWordGate:
    """Gates mic audio behind a wake word; starts open, toggles closed/open on detection."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        threshold: float,
        on_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        """Load the wake-word model (if enabled) and start with the gate already open."""
        self._threshold = threshold
        self._on_toggle = on_toggle
        self._model_name = model_name
        self._model: object | None = None
        self._buffer: NDArray[np.int16] = np.empty((0,), dtype=np.int16)
        self._last_toggle_time = 0.0
        # A single utterance produces several consecutive above-threshold predictions; only
        # toggle on the rising edge (first frame that crosses the threshold), not every frame.
        self._above_threshold = False
        # Reachy starts awake and listening; saying the wake word puts it to sleep, saying it
        # again wakes it back up. Also fails open if the model can't load.
        self.active = True

        if enabled:
            self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> object | None:
        try:
            import openwakeword
            from openwakeword.utils import download_models

            download_models([model_name])
            model: object = openwakeword.Model(wakeword_models=[model_name], inference_framework="onnx")
            return model
        except Exception as e:
            logger.warning("Wake word model %r unavailable (%s); wake word gate disabled.", model_name, e)
            return None

    def process(self, sample_rate: int, audio_frame: NDArray[np.int16]) -> None:
        """Feed one mic frame through the detector, toggling activation when the wake word is heard."""
        if self._model is None:
            return

        self._buffer = np.concatenate([self._buffer, self._resample_to_target(sample_rate, audio_frame)])
        while len(self._buffer) >= _CHUNK_SAMPLES:
            chunk, self._buffer = self._buffer[:_CHUNK_SAMPLES], self._buffer[_CHUNK_SAMPLES:]
            self._detect(chunk)

    @staticmethod
    def _resample_to_target(sample_rate: int, audio_frame: NDArray[np.int16]) -> NDArray[np.int16]:
        if audio_frame.ndim > 1:
            audio_frame = audio_frame[:, 0]
        if sample_rate == TARGET_SAMPLE_RATE:
            return audio_frame.astype(np.int16, copy=False)

        target_length = round(len(audio_frame) * TARGET_SAMPLE_RATE / sample_rate)
        if target_length <= 0:
            return np.empty((0,), dtype=np.int16)
        resampled: NDArray[np.int16] = resample(audio_frame, target_length).astype(np.int16)
        return resampled

    def _detect(self, chunk: NDArray[np.int16]) -> None:
        try:
            predictions = self._model.predict(chunk)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Wake word inference failed (%s); disabling wake word gate.", e)
            self._model = None
            self.active = True
            return

        score = predictions.get(self._model_name, 0.0)
        if score < self._threshold:
            self._above_threshold = False
            return
        if self._above_threshold:
            # Still the same utterance as the frame that already triggered the toggle.
            return
        self._above_threshold = True

        now = time.monotonic()
        if (now - self._last_toggle_time) < _REFRACTORY_SECONDS:
            return

        self._last_toggle_time = now
        self.active = not self.active
        logger.info("Wake word detected (score=%.2f); gate now %s.", score, "active" if self.active else "inactive")
        if self._on_toggle is not None:
            try:
                self._on_toggle(self.active)
            except Exception:
                logger.warning("Wake word toggle callback raised (ignored)", exc_info=True)
