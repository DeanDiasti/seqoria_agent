#!/usr/bin/env python
"""
Stream TTS with ElevenLabsTTS and play it out loud using the built-in
`SpeakerSink` helper (which internally wraps *sounddevice*).

Prerequisites
-------------
pip install "seqoria-agent[elevenlabs,speech]"
export ELEVENLABS_API_KEY="<your_key>"
"""

from __future__ import annotations
import asyncio
# Replace direct sounddevice usage with SpeakerSink which is part of the speech_client package.
from seqoria_agent.tts_providers import ElevenLabsTTS
from seqoria_agent.client.speech_client.audio.sink import SpeakerSink
 
VOICE = "Xb7hH8MSUJpSbSDYk0k2"
MODEL = "eleven_flash_v2_5"
TEXT  = (
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

async def text_iter() -> asyncio.AsyncIterator[str]:
    await asyncio.sleep(1.5)
    print("Sending text")
    yield TEXT
    await asyncio.sleep(1.5)
    print("Sending text")
    yield TEXT
    await asyncio.sleep(1.5)
    print("Sending text")
    yield TEXT

async def main() -> None:
    tts = ElevenLabsTTS(voice=VOICE, model=MODEL)

    sink = SpeakerSink(sample_rate=24_000, channels=1)

    async for chunk in tts.stream(text_iter()):
        await sink.consume(chunk)

    # Graceful shutdown
    await sink.close()

if __name__ == "__main__":
    asyncio.run(main())