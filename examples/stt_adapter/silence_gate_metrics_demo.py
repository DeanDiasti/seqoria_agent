"""Interactive SilenceGate metrics demo.

This script captures audio from the system's default microphone using
:class:`seqoria_agent.client.speech_client.audio.source.MicSource`, feeds the
frames through :class:`seqoria_agent.client.speech_client.utils.SilenceGate`
so we can detect *end-of-utterance* events, and finally hands the buffered
utterance to :class:`seqoria_agent.stt_providers.FireworksSTT` for
transcription.

Metrics such as utterance duration and speech-to-text latency are printed
whenever SilenceGate signals that an utterance has ended.  The loop continues
indefinitely until interrupted (Ctrl+C).

Requirements
------------
1. ``sounddevice`` Python package (install via
   ``pip install \"seqoria-agent[speech]\"``).
2. ``fireworks-ai`` Python package (``pip install fireworks-ai``).
3. ``ffmpeg`` command-line tool available in ``PATH``.
4. ``FIREWORKS_API_KEY`` environment variable set.

Run::

    python -m examples.stt_adapter.silence_gate_metrics_demo

You can adjust gate behaviour with CLI flags::

    python -m examples.stt_adapter.silence_gate_metrics_demo \\
        --threshold-ms 800 --energy-gate 35 --chunk-ms 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import List

from dotenv import load_dotenv

from seqoria_agent.client.speech_client.audio.source import MicSource
from seqoria_agent.client.speech_client.utils import MovingAverageTimer, SilenceGate
from seqoria_agent.models.speech_models import AudioChunk
from seqoria_agent.stt_providers import FireworksSTT

load_dotenv()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _print_transcripts(chunks: List):  # noqa: D401 – pretty printer helper
    for t in chunks:
        role = "FINAL" if t.is_final else "PART "
        print(f"    [{t.start_ts:7.2f}-{t.end_ts:7.2f}] {role}: {t.text}")


async def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-ms", type=int, default=600,
                        help="Trailing silence (ms) required to end an utterance")
    parser.add_argument("--energy-gate", type=float, default=30.0,
                        help="Frames below this dBFS considered silent")
    parser.add_argument("--chunk-ms", type=int, default=20,
                        help="Microphone frame size in milliseconds")
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    gate = SilenceGate(threshold_ms=args.threshold_ms, energy_gate=args.energy_gate)
    mic = MicSource(sample_rate=16_000, chunk_ms=args.chunk_ms)
    stt = FireworksSTT(api_key=api_key, overlap=750, chunk_duration=2000)

    latency_timer = MovingAverageTimer()

    utterance_buf = bytearray()
    utterance_start: float | None = None

    print("🎤  Speak into the microphone - Ctrl+C to exit\n")

    try:
        async for frame in mic.frames():
            fired = gate.update(frame)

            # Accumulate audio while speech detected
            if gate.in_speech:
                if utterance_start is None:
                    utterance_start = time.monotonic()
                    print("▶️  Speech detected - recording…")
                utterance_buf.extend(frame.pcm)

            if fired and utterance_start is not None:
                # ------------------------------------------------------------------
                # End-of-utterance reached – transcribe buffered audio
                # ------------------------------------------------------------------
                duration_ms = (time.time() - utterance_start) * 1000.0
                print(f"⏹️  Utterance ended - duration: {duration_ms:.0f} ms,\n    bytes: {len(utterance_buf)}")

                t0 = time.time()
                transcripts = await stt.transcribe(pcm=bytes(utterance_buf),
                                                    sample_rate=16_000,
                                                    channels=1)
                latency = time.time() - t0
                latency_timer.add(latency)

                print(f"📝  Transcripts ({len(transcripts)} chunks, latency {latency:.2f}s):")
                _print_transcripts(transcripts)

                print(f"📊  Latency stats - count: {latency_timer.count}, "
                      f"p50: {latency_timer.p50:.2f}s, p95: {latency_timer.p95:.2f}s, "
                      f"avg: {latency_timer.avg:.2f}s\n")

                # Reset for next utterance
                utterance_buf.clear()
                utterance_start = None
                gate.reset()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    asyncio.run(main()) 