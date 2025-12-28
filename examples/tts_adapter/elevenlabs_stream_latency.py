#!/usr/bin/env python
"""Example: Measure latency when streaming TTS using the ElevenLabsTTS adapter.

Prerequisites:
    pip install seqoria-agent[elevenlabs]  # installs websockets + httpx extras
    export ELEVENLABS_API_KEY="<your_api_key>"

This script streams a large paragraph of text to the adapter and prints timing
information for:
    • time to first audio chunk (initial latency)
    • total streaming duration
    • number of chunks and total PCM bytes received

Run via:
    python examples/tts_adapter/elevenlabs_stream_latency.py
"""
from __future__ import annotations

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()
from seqoria_agent.tts_providers import ElevenLabsTTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use a sizable paragraph to ensure multiple chunks are generated.
TEXT_TO_SYNTHESIZE = (
    "I have a dream that one day this nation will rise up and live out the true meaning of its creed: "
    "'We hold these truths to be self-evident, that all men are created equal.' "
    "I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners "
    "will be able to sit down together at the table of brotherhood. "
    "I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, "
    "sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice. "
    "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin "
    "but by the content of their character. "
    "I have a dream today!"
)

VOICE = os.getenv("ELEVENLABS_VOICE", "Xb7hH8MSUJpSbSDYk0k2")  # default voice
MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def text_iter() -> asyncio.AsyncIterator[str]:  # type: ignore[override]
    """Yield the entire text once, ensuring terminating space per protocol."""
    yield TEXT_TO_SYNTHESIZE + " "


async def main() -> None:
    tts = ElevenLabsTTS(voice=VOICE, model=MODEL)
    print("Starting ElevenLabs TTS streaming test…")

    start_ts = time.perf_counter()
    first_chunk_ts: float | None = None
    total_bytes = 0
    chunk_count = 0

    async for chunk in tts.stream(text_iter()):
        now = time.perf_counter()
        if first_chunk_ts is None:
            first_chunk_ts = now
            print(f"First audio chunk received after {first_chunk_ts - start_ts:.3f}s")
        chunk_count += 1
        total_bytes += len(chunk.pcm)

    end_ts = time.perf_counter()
    print("--- Summary ---")
    print(f"Chunks received : {chunk_count}")
    print(f"Total PCM bytes : {total_bytes}")
    print(f"Total time      : {end_ts - start_ts:.3f}s")
    if first_chunk_ts is not None:
        print(f"First→End delta : {end_ts - first_chunk_ts:.3f}s")


if __name__ == "__main__":
    asyncio.run(main()) 