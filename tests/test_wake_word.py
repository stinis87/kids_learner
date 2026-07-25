"""Unit tests for the local wake-word gate."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from reachy_mini_conversation_app.wake_word import _CHUNK_SAMPLES, WakeWordGate


class _FakeModel:
    """A stub openWakeWord model returning a scripted sequence of scores."""

    def __init__(self, model_name: str, scores: list[float]) -> None:
        self._model_name = model_name
        self._scores = iter(scores)

    def predict(self, _chunk: np.ndarray) -> dict[str, float]:
        return {self._model_name: next(self._scores, 0.0)}


def _make_gate(scores: list[float], *, threshold: float = 0.5, on_toggle: Any = None) -> WakeWordGate:
    model_name = "hey_jarvis"
    with patch.object(WakeWordGate, "_load_model", return_value=_FakeModel(model_name, scores)):
        return WakeWordGate(enabled=True, model_name=model_name, threshold=threshold, on_toggle=on_toggle)


def _feed_chunk(gate: WakeWordGate) -> None:
    """Feed exactly one 1280-sample chunk at the gate's native 16kHz rate."""
    gate.process(16000, np.zeros(_CHUNK_SAMPLES, dtype=np.int16))


def test_disabled_gate_is_always_active_and_does_not_load_a_model() -> None:
    """A disabled gate stays open and never touches the model loader."""
    gate = WakeWordGate(enabled=False, model_name="hey_jarvis", threshold=0.5)
    assert gate.active is True
    gate.process(16000, np.zeros(_CHUNK_SAMPLES, dtype=np.int16))
    assert gate.active is True


def test_missing_model_fails_open() -> None:
    """If the model can't load, the gate fails open instead of silently ignoring everyone."""
    with patch.object(WakeWordGate, "_load_model", return_value=None):
        gate = WakeWordGate(enabled=True, model_name="hey_jarvis", threshold=0.5)
    assert gate.active is True


def test_gate_starts_closed_and_opens_on_detection() -> None:
    """The gate starts closed and only opens once the wake word scores above threshold."""
    gate = _make_gate([0.1, 0.9])
    assert gate.active is False

    _feed_chunk(gate)  # score 0.1, below threshold
    assert gate.active is False

    _feed_chunk(gate)  # score 0.9, triggers activation
    assert gate.active is True


def test_second_detection_toggles_gate_closed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saying the wake word again toggles the gate back closed."""
    toggles: list[bool] = []
    gate = _make_gate([0.9, 0.9], on_toggle=toggles.append)

    # Force the refractory window closed so the second trigger below is honored immediately.
    monkeypatch.setattr("reachy_mini_conversation_app.wake_word._REFRACTORY_SECONDS", 0.0)

    _feed_chunk(gate)
    assert gate.active is True
    assert toggles == [True]

    _feed_chunk(gate)
    assert gate.active is False
    assert toggles == [True, False]


def test_refractory_period_prevents_double_toggle_from_one_utterance() -> None:
    """Repeated high scores from a single utterance must not toggle the gate twice."""
    gate = _make_gate([0.9, 0.9, 0.9])

    _feed_chunk(gate)
    assert gate.active is True

    # Immediately repeated high scores (same utterance trailing on) must not re-toggle.
    _feed_chunk(gate)
    _feed_chunk(gate)
    assert gate.active is True


def test_inference_failure_disables_gate_and_fails_open() -> None:
    """A crashing model disables itself and fails open rather than breaking the conversation loop."""

    class _RaisingModel:
        def predict(self, _chunk: np.ndarray) -> dict[str, float]:
            raise RuntimeError("boom")

    with patch.object(WakeWordGate, "_load_model", return_value=_RaisingModel()):
        gate = WakeWordGate(enabled=True, model_name="hey_jarvis", threshold=0.5)
    assert gate.active is False

    _feed_chunk(gate)
    assert gate.active is True
