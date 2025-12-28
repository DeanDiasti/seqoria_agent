"""Interactive microphone → SpeechClient demo.

Requires optional extras: pip install "seqoria-agent[speech,whisper,elevenlabs]"
Run: python mic_loop.py --duration 30

This CLI showcases the **hybrid end-of-utterance detection** added in Phase 10-12.
Tuning flags:
  --vad-timeout-ms   trailing-silence (audio) threshold before a turn ends
  --stt-quiet-ms     STT inactivity (no new transcripts) threshold

See docs/guides/speech_client.md for recommendations.
"""

from __future__ import annotations

import argparse
import asyncio

from seqoria_agent.client.speech_client import SpeechClient
from seqoria_agent.client.speech_client.audio import MicSource, SpeakerSink
from seqoria_agent.stt_providers import FireworksSTT
from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS
from seqoria_agent.client.ai_client import AIClient
from seqoria_agent.llm_providers.openai_provider import OpenAIProvider


async def main():  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30,
                        help="Max conversation duration in seconds (demo safety cutoff)")
    parser.add_argument("--vad-timeout-ms", type=int, default=600,
                        help="Milliseconds of trailing silence (audio level) to detect end-of-utterance")
    parser.add_argument("--stt-quiet-ms", type=int, default=800,
                        help="Milliseconds with no new STT tokens before forcing turn end")
    args = parser.parse_args()

    ai = AIClient(providers=[OpenAIProvider()])
    speech = SpeechClient(ai_client=ai, 
                          source=MicSource(), 
                          stt=FireworksSTT(), 
                          tts=ElevenLabsTTS(), 
                          sink=SpeakerSink(), 
                          vad_timeout_ms=args.vad_timeout_ms, 
                          stt_quiet_ms=args.stt_quiet_ms)

    async with ai, speech:
        async def _runner():
            async for _ in speech.stream_converse():
                pass

        await asyncio.wait_for(_runner(), timeout=args.duration)


if __name__ == "__main__":
    asyncio.run(main()) 