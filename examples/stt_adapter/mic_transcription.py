from __future__ import annotations

"""Demonstration of real-time microphone transcription with FireworksSTT.

This script captures audio from the system's default microphone using
:class:`seqoria_agent.client.speech_client.audio.source.MicSource` and feeds
frames to :class:`seqoria_agent.stt_providers.FireworksSTT`. Partial
transcripts are printed to the console as they arrive.

Requirements
------------
1. ``sounddevice`` Python package (install via ``pip install \"seqoria-agent[speech]\"``).
2. ``fireworks-ai`` Python package (``pip install fireworks-ai``).
3. ``FIREWORKS_API_KEY`` environment variable set.

Run::

    python -m examples.stt_adapter.fireworks_mic_streaming
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from seqoria_agent.client.speech_client.audio.source import MicSource
from seqoria_agent.stt_providers import FireworksSTT

load_dotenv()


aSYNC_FRAME_MS = 20  # Microphone frame size in milliseconds


async def main() -> None:  # noqa: D401
    """Capture microphone audio and print incremental transcripts."""

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    mic = MicSource(sample_rate=16_000, chunk_ms=aSYNC_FRAME_MS)

    print("Starting microphone transcription…  (Ctrl+C to stop)")

    async with FireworksSTT(api_key=api_key, overlap=750, chunk_duration=2000) as stt:
        try:
            async for chunk in stt.stream(frames=mic.frames()):
                role = "FINAL" if chunk.is_final else "PART "
                print(f"[{chunk.start_ts:7.2f}-{chunk.end_ts:7.2f}] {role}: {chunk.text}")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")


if __name__ == "__main__":
    asyncio.run(main()) 