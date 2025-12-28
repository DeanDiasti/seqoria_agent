"""WebRTC remote speech demo using RTCSource/RTCSink (simplified)."""

from __future__ import annotations

import argparse
import asyncio

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import ApprtcSignaling  # example signalling

from seqoria_agent.client.speech_client import SpeechClient
from seqoria_agent.client.speech_client.audio import SpeakerSink
from seqoria_agent.client.speech_client.audio.rtc_source import RTCSource
from seqoria_agent.client.speech_client.audio.rtc_sink import RTCSink
from seqoria_agent.stt_providers import FireworksSTT
from seqoria_agent.tts_providers import ElevenLabsTTS
from seqoria_agent.client.ai_client import AIClient
from seqoria_agent.llm_providers.openai_provider import OpenAIProvider


async def run(signalling_url: str):  # noqa: D401
    signaling = ApprtcSignaling(signalling_url)
    pc = RTCPeerConnection()

    await signaling.connect()

    @pc.on("track")
    def on_track(track):  # noqa: D401
        if track.kind == "audio":
            asyncio.create_task(handle_audio_track(track))

    # Exchange SDP ----------------------------------------------
    await pc.setLocalDescription(await pc.createOffer())
    await signaling.send(pc.localDescription)
    answer = await signaling.receive()
    await pc.setRemoteDescription(answer)

    # keep running
    await asyncio.Future()


async def handle_audio_track(track):  # noqa: D401
    ai = AIClient(providers=[OpenAIProvider()])
    speech = SpeechClient(
        ai_client=ai,
        source=RTCSource(track),
        stt=FireworksSTT(),
        tts=ElevenLabsTTS(),
        sink=SpeakerSink(),
    )
    async with ai, speech:
        async for _ in speech.stream_converse():
            pass


async def main():  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--signalling", required=True)
    args = parser.parse_args()
    await run(args.signalling)


if __name__ == "__main__":
    asyncio.run(main()) 