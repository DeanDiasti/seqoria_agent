# SpeechClient Guide

The `SpeechClient` package turns raw audio into conversational AI magic in **≈10 lines of code**.

It hides gritty streaming-details (microphones, WebRTC, VAD, STT, LLM, TTS) yet stays fully *hackable* – you can swap adapters, inspect latency metrics, even craft your own loops.

---
## 1 Architecture (20 000 ft)

```mermaid
flowchart TD
    A[AudioSource\n(mic / file / RTC)] -->|AudioChunk| B[SpeechPipeline]
    B -->|frames→STT| C[STT Adapter]
    C -->|TranscriptChunk| B
    subgraph Pipeline Internals
        B -->|user text| D[SpeechClient]
        D --> E[AIClient]
        E -->|assistant text| B
        B -->|text→TTS| F[TTS Adapter]
        F -->|AudioChunk| B
    end
    B -->|AudioChunk| G[AudioSink\n(speaker / RTC)]
```

*Each rectangle is a **replaceable plug-in**.*  The pipeline glues them together with asyncio queues.

---
## 2 Quick Start – 2 Liner

```python
from seqoria_agent.client.speech_client import SpeechClient
from seqoria_agent.client.speech_client.audio import MicSource, SpeakerSink
from seqoria_agent.stt_providers import WhisperAdapter
from seqoria_agent.tts_providers import ElevenLabsAdapter
from seqoria_agent import AIClient

async with SpeechClient(ai_client=AIClient(),
                         source=MicSource(),
                         stt=WhisperAdapter(),
                         tts=ElevenLabsAdapter(),
                         sink=SpeakerSink()) as sc:
    await sc.stream_converse()  # low-latency streaming both ways
```

Install speech extras:  `pip install "seqoria-agent[speech]"`  (pulls *sounddevice*, *whisperx*, *aiortc*, …).

---
## 3 Public Helpers

| Attribute | Purpose |
|-----------|---------|
| `pipeline` | Direct access to internal `SpeechPipeline` (queues, start/stop). |
| `metrics`  | Rolling latency stats (p50/p95/p99) + counters – ideal for dashboards. |
| `history`  | Mutable chat history – prepend system messages or log turns. |
| `on(event, cb)` | Subscribe to non-blocking callbacks.  Built-in events:<br>• `user_start / user_stop`<br>• `assistant_start / assistant_stop`<br>• `vad_trigger` – silence budget reached<br>• `stt_timeout` – no new tokens inside `stt_quiet_ms` |

All callbacks run in the background – keep them lightweight.

---
## 4 Convenience Loops

1. **`stream_converse()`**  – duplex stream (STT → AI → TTS) forever until the AudioSource ends.
2. **`converse_once()`**   – batch helper:

```python
pcm_reply = await sc.converse_once(wav_bytes=some_blob)
```

It transcribes the utterance, runs a single LLM turn, generates one TTS reply, plays it through the sink and finally returns the PCM bytes.

---
## 5 Rolling Your Own Loop

Need exotic cut-off rules or UI integration?  Use the raw pipeline queues:

```python
async for t in sc.pipeline.transcript_q_iter():
    if my_custom_condition(t):
        assistant_text = await sc._agentic_once(t.text)
        await sc.pipeline.tts_text_q.put(assistant_text)
```

---
## 6 Latency Metrics Example

```python
print(f"p50 latency → {sc.metrics.latency.p50*1000:.0f} ms")
```

Metrics update automatically after every assistant turn – no polling required.

---
## 7 Event Logging Snippet

```python
sc.on("assistant_start", lambda p: print("assistant speaking…"))
sc.on("vad_trigger",      lambda p: print("silence ended user turn"))
```

---
## 8 What’s Next?

* **Examples** – see `examples/voice_chat/manual_speech_agent.py` for a full customise-everything demo.
* **Docs** – more details under *docs/topics/* (VAD, streaming, error-handling).

Happy hacking! 🎙️🤖 