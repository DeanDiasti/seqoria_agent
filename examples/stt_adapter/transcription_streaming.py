from __future__ import annotations

"""Demonstration of pseudo-streaming transcription with FireworksSTT.

This script pulls a live MP3 stream from the BBC World Service, decodes it to
16-kHz mono PCM via *ffmpeg*, and feeds the audio frames to
:class:`seqoria_agent.stt_providers.FireworksSTT`.  Partial transcripts are
printed to the console as they arrive.

Requirements
------------
1. ``ffmpeg`` binary available on your ``$PATH`` for audio decoding.
2. ``fireworks-ai`` Python package installed (``pip install fireworks-ai``).
3. ``FIREWORKS_API_KEY`` environment variable set.

Run::

    python -m examples.stt_adapter.fireworks_streaming
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from seqoria_agent.stt_providers import FireworksSTT

load_dotenv()

# Live BBC World Service MP3 stream – replace with any radio URL of your choice.
STREAM_URL = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"


async def main() -> None:  # noqa: D401
    """Stream live radio and print incremental transcripts."""

    # Allow configuration via CLI environment variables
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print("Starting live transcription…  (Ctrl+C to stop)")

    async with FireworksSTT(api_key=api_key, chunk_duration=1000, overlap=250) as stt:
        try:
            async for chunk in stt.stream(url=STREAM_URL):
                role = "FINAL" if chunk.is_final else "PART"

                print(f"[{chunk.start_ts:7.2f}-{chunk.end_ts:7.2f}] {role}: {chunk.text}")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")


if __name__ == "__main__":
    asyncio.run(main())
