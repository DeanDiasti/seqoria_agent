"""Utility tests for MovingAverageTimer & SilenceGate."""

from __future__ import annotations

import pytest

from seqoria_agent.client.speech_client.utils import MovingAverageTimer, SilenceGate
from seqoria_agent.models.speech_models import AudioChunk

_SAMPLE_RATE = 16_000
_FRAME_MS = 20


def _chunk(value: int) -> AudioChunk:
    samples = int(_SAMPLE_RATE * _FRAME_MS / 1000)
    pcm = value.to_bytes(2, "little", signed=True) * samples
    return AudioChunk(pcm=pcm, sample_rate=_SAMPLE_RATE, channels=1)


def test_moving_average_timer():  # noqa: D401
    timer = MovingAverageTimer(window=4)
    for v in [0.1, 0.2, 0.3, 0.4]:
        timer.add(v)
    assert round(timer.avg, 2) == 0.25
    assert timer.count == 4
    assert timer.p50 > 0


def test_silence_gate_detection():  # noqa: D401
    gate = SilenceGate(threshold_ms=600)

    # Leading silence should *not* trigger.
    for _ in range(5):  # 100 ms silence
        assert gate.update(_chunk(0)) is False

    # Speech frames reset gate.
    for _ in range(2):
        assert gate.update(_chunk(10000)) is False

    # 600 ms trailing silence should trigger.
    triggered = False
    for _ in range(30):
        if gate.update(_chunk(0)):
            triggered = True
            break
    assert triggered, "SilenceGate did not detect end-of-utterance" 