"""Tests for the kid-friendly robotic voice effect."""

from __future__ import annotations

import numpy as np

from reachy_mini_conversation_app.audio.voice_effect import (
    VoiceEffectState,
    apply_kid_robot_voice_effect,
)


SAMPLE_RATE = 16000


def _tone(frequency_hz: float, n_samples: int, amplitude: int = 16000) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.int16).reshape(1, -1)


def test_no_op_passthrough_when_disabled() -> None:
    """A disabled effect returns the original array unchanged."""
    pcm = _tone(220.0, 800)
    out = apply_kid_robot_voice_effect(
        pcm,
        SAMPLE_RATE,
        state=VoiceEffectState(),
        pitch_factor=1.0,
        robot_mix=0.0,
        robot_carrier_hz=60.0,
    )
    assert out is pcm


def test_pitch_factor_shortens_chunk() -> None:
    """Pitching up resamples the chunk to fewer samples at the same rate."""
    pcm = _tone(220.0, 1600)
    out = apply_kid_robot_voice_effect(
        pcm,
        SAMPLE_RATE,
        state=VoiceEffectState(),
        pitch_factor=1.25,
        robot_mix=0.0,
        robot_carrier_hz=60.0,
    )
    assert out.shape[1] == round(1600 / 1.25)


def test_ring_modulation_stays_in_int16_range() -> None:
    """Ring modulation alters the signal but never overflows int16."""
    pcm = _tone(220.0, 1600, amplitude=32767)
    out = apply_kid_robot_voice_effect(
        pcm,
        SAMPLE_RATE,
        state=VoiceEffectState(),
        pitch_factor=1.0,
        robot_mix=0.2,
        robot_carrier_hz=60.0,
    )
    assert out.dtype == np.int16
    assert np.all(out <= 32767) and np.all(out >= -32768)
    assert not np.array_equal(out, pcm)


def test_phase_continuity_across_chunks() -> None:
    """The carrier phase must carry over between chunks to avoid boundary clicks."""
    state = VoiceEffectState()
    pcm = _tone(220.0, 400)
    first = apply_kid_robot_voice_effect(
        pcm, SAMPLE_RATE, state=state, pitch_factor=1.0, robot_mix=0.2, robot_carrier_hz=60.0
    )
    phase_after_first = state.carrier_phase
    second = apply_kid_robot_voice_effect(
        pcm, SAMPLE_RATE, state=state, pitch_factor=1.0, robot_mix=0.2, robot_carrier_hz=60.0
    )
    # A fresh state restarting the carrier from zero would reproduce the first
    # chunk's output; the continuous state must diverge because the carrier
    # phase kept advancing.
    assert phase_after_first != 0.0
    assert not np.array_equal(first, second)


def test_empty_array_does_not_crash() -> None:
    """An empty chunk is a safe no-op."""
    pcm = np.zeros((1, 0), dtype=np.int16)
    out = apply_kid_robot_voice_effect(
        pcm,
        SAMPLE_RATE,
        state=VoiceEffectState(),
        pitch_factor=1.25,
        robot_mix=0.2,
        robot_carrier_hz=60.0,
    )
    assert out.size == 0
