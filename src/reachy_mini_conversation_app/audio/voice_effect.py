"""Cheap, real-time DSP effect that turns the TTS voice into a higher-pitched, robotic one.

Runs on every outgoing audio chunk, so it must stay O(n) with no FFT or model
inference: a linear-interpolation resample raises the pitch (the classic
"chipmunk" trick), and a phase-continuous ring modulation adds the robotic
character.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VoiceEffectState:
    """Carries the ring-modulator carrier phase across chunks to avoid clicks."""

    carrier_phase: float = 0.0

    def reset(self) -> None:
        """Reset the carrier phase, e.g. at the start of a new session."""
        self.carrier_phase = 0.0


def apply_kid_robot_voice_effect(
    pcm: NDArray[np.int16],
    sample_rate: int,
    *,
    state: VoiceEffectState,
    pitch_factor: float,
    robot_mix: float,
    robot_carrier_hz: float,
) -> NDArray[np.int16]:
    """Return `pcm` pitched up and lightly ring-modulated for a childish, robotic voice.

    `pcm` is expected as a `(1, n_samples)` int16 array, matching the realtime
    handler's audio delta shape. A `pitch_factor` of 1.0 and `robot_mix` of 0.0
    are no-ops (returns `pcm` unchanged).
    """
    if pcm.size == 0 or (pitch_factor == 1.0 and robot_mix == 0.0):
        return pcm

    samples = pcm.reshape(-1).astype(np.float32)
    shifted = _pitch_shift_up(samples, pitch_factor) if pitch_factor != 1.0 else samples
    robotic = _ring_modulate(shifted, sample_rate, robot_carrier_hz, robot_mix, state) if robot_mix > 0.0 else shifted

    return np.clip(robotic, -32768, 32767).astype(np.int16).reshape(1, -1)


def _pitch_shift_up(samples: NDArray[np.float32], pitch_factor: float) -> NDArray[np.float32]:
    """Resample to fewer points at the same sample rate, raising pitch and speed together."""
    n_in = samples.shape[0]
    n_out = max(1, round(n_in / pitch_factor))
    source_index = np.linspace(0, n_in - 1, num=n_out, dtype=np.float32)
    return np.interp(source_index, np.arange(n_in, dtype=np.float32), samples).astype(np.float32)


def _ring_modulate(
    samples: NDArray[np.float32],
    sample_rate: int,
    carrier_hz: float,
    mix: float,
    state: VoiceEffectState,
) -> NDArray[np.float32]:
    """Mix in a carrier-modulated copy of the signal for a robotic timbre."""
    n = samples.shape[0]
    phase_step = 2.0 * math.pi * carrier_hz / sample_rate
    phases = state.carrier_phase + phase_step * np.arange(n, dtype=np.float32)
    state.carrier_phase = float((state.carrier_phase + phase_step * n) % (2.0 * math.pi))
    carrier = np.cos(phases, dtype=np.float32)
    return (1.0 - mix) * samples + mix * samples * carrier
