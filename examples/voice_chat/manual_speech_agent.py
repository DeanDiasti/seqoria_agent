#!/usr/bin/env python3
"""Interactive speech agent – customise everything.

Run with a microphone (default) or a WAV file:

```bash
python examples/voice_chat/manual_speech_agent.py               # mic
python examples/voice_chat/manual_speech_agent.py --wav demo.wav
```

Requirements
------------
• `pip install "seqoria-agent[speech]"`
• For real STT/TTS provide your own adapters (the demo stubs just echo text).
"""
from __future__ import annotations

import asyncio, time, os
from dotenv import load_dotenv

from seqoria_agent.client.speech_client.audio import MicSource, SpeakerSink
from seqoria_agent import AIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.tts_providers import ElevenLabsTTS
from seqoria_agent.stt_providers import FireworksSTT
from seqoria_agent.llm_providers import OpenAIProvider
from seqoria_agent.client.speech_client.utils import SilenceGate
from seqoria_agent.models.llm_models import ChatChunk, AssistantMessage, ToolCallRequestList

load_dotenv()


tools = LocalToolbox()

@tools.tool(description="Return sunny weather always")
def weather(city: str) -> str:  # noqa: D401 – toy example
    return f"It is always sunny in {city}."



def _log(evt: str):  # noqa: D401 – tiny util
    def _inner(payload):
        ts = payload.get("timestamp", time.time())
        preview = payload.get("partial", payload.get("preview", ""))
        print(f"[{ts:.3f}] {evt}: {preview}")

    return _inner



async def main():  # noqa: D401 – entrypoint
    conversation_history = []


    sink = SpeakerSink(sample_rate=24_000)

    stt = FireworksSTT(api_key=os.getenv("FIREWORKS_API_KEY"))
    tts = ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"), batch_words=15)
    
    gate = SilenceGate(threshold_ms=600, energy_gate=30.0)

    async with AIClient(providers=[OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")], local_toolbox=tools) as ai:
        while True:
            utterance_start = None
            agg_buf = bytearray()  # Buffer for real-time 2-second chunks
            SAMPLE_RATE = 16_000
            # Create a fresh microphone source for *this* user turn so we
            # don't capture assistant playback or residual frames.
            source = MicSource(sample_rate=SAMPLE_RATE, chunk_ms=20)
            BYTES_PER_SAMPLE = 2  # 16-bit PCM
            CLIP_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 2  # ≈ 64 KB for 2-second chunk

            gate.reset()
            _ = input("enter to speak now - Ctrl-C to quit…")

            try:
                user_query = ""

                # get user spoken input
                async for frame in source.frames():
                    fired = gate.update(frame)

                    if gate.in_speech:
                        if utterance_start is None:
                            utterance_start = time.monotonic()
                            print("Speech detected - recording...")
                        # Aggregate PCM for both real-time chunking *and* full utterance buffer
                        agg_buf.extend(frame.pcm)

                        # Once we have ~2 seconds of audio → send to STT immediately
                        if len(agg_buf) >= CLIP_BYTES:
                            transcript = await stt.transcribe(pcm=bytes(agg_buf),
                                                            sample_rate=16_000,
                                                            channels=1)
                            user_query += transcript[0].text
                            print(transcript[0].text, end=" ", flush=True)
                            agg_buf.clear()

                    if fired and utterance_start is not None:
                        # Transcribe any residual audio (<2 s) that hasn't been sent yet
                        end_time = time.monotonic()
                        if agg_buf:
                            residual = await stt.transcribe(pcm=bytes(agg_buf),
                                                            sample_rate=16_000,
                                                            channels=1)
                            user_query += residual[0].text
                            print(residual[0].text, end=" ", flush=True)
                            agg_buf.clear()
                        latency = time.monotonic() - end_time
                        print(f"\ntranscription ended - duration: {latency:.0f} ms")
                        break
            
                # send user query to LLM
                conversation_history.append({"role": "user", "content": user_query})

                print("\nAssistant:")
                # ------------------------------------------------------------------
                # Queue-based text streaming → ElevenLabsTTS.stream
                # ------------------------------------------------------------------
                text_q: asyncio.Queue[str | None] = asyncio.Queue()

                async def _text_iter():  # noqa: D401 – yields strings from queue
                    while True:
                        txt = await text_q.get()
                        if txt is None:
                            break
                        yield txt

                async def _playback():  # noqa: D401 – consume audio chunks
                    async for audio in tts.stream(text_iter=_text_iter()):
                        await sink.consume(audio)


                playback_task = asyncio.create_task(_playback())


                sent_so_far = ""
                async for chunk in ai.llm_stream_once(history=conversation_history):
                    if isinstance(chunk, ChatChunk):
                        delta = chunk.content[len(sent_so_far):] if chunk.content.startswith(sent_so_far) else chunk.content
                        sent_so_far = chunk.content
                        print(delta, end="", flush=True)
                        await text_q.put(delta)
                        continue

                    if isinstance(chunk, AssistantMessage):
                        await text_q.put(None)  # signal end of utterance
                        await playback_task  # wait until audio done
                        conversation_history.append({"role": "assistant", "content": chunk.content})
                        break  # assistant turn finished

                    if isinstance(chunk, ToolCallRequestList):
                        print("\tTool calls:")
                        async for req, res in ai.run_call_stream(tool_calls=chunk):
                            conversation_history.extend([req, res])
                            print(f"\t\t• {req.name} (id={req.id}) → {res.result}")
                
                print("\n")
    
            except KeyboardInterrupt:
                print("\nExiting.")



if __name__ == "__main__":  # pragma: no cover – CLI entry
    asyncio.run(main()) 