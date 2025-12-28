"""Simple push-to-talk voice chat server for Seqoria-Agent examples.

Run with:
    pip install "seqoria-agent[speech,whisper,elevenlabs]" fastapi uvicorn python-multipart

    export ELEVENLABS_API_KEY=...   # required for ElevenLabsTTS
    export FIREWORKS_API_KEY=...    # required for FireworksSTT (or switch to WhisperSTT)

    uvicorn voice_server:app --reload

POST /api/speech with a multipart field named *file* containing a short
microphone recording (e.g. WebM or WAV).  The endpoint returns the
transcribed user text and the assistant reply synthesised by ElevenLabs
as base-64 WAV audio for easy playback in the browser.
"""
from __future__ import annotations

import base64
import tempfile
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware  # NEW
from pydantic import BaseModel

from seqoria_agent.client.ai_client import AIClient
from seqoria_agent.llm_providers.openai_provider import OpenAIProvider
from seqoria_agent.stt_providers.fireworks_stt import FireworksSTT
from seqoria_agent.tts_providers.elevenlabs_adapter import ElevenLabsTTS
from seqoria_agent.client.speech_client import SpeechClient
from seqoria_agent.client.speech_client.audio.source import FileSource
from seqoria_agent.client.speech_client.audio.sink import BaseAudioSink
from seqoria_agent.models.speech_models import AudioChunk

app = FastAPI(title="Seqoria Voice Chat Demo")

# Allow requests from any origin (simplifies local testing – narrow in production)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

# ---------------------------------------------------------------------------
# Helper sink – collects TTS audio into memory
# ---------------------------------------------------------------------------
class _BufferSink(BaseAudioSink):
    """Accumulates AudioChunk PCM data so we can package it in the response."""

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()

    async def consume(self, chunk: AudioChunk) -> None:  # noqa: D401
        self._buf.extend(chunk.pcm)

    @property
    def pcm(self) -> bytes:  # 16-bit little-endian PCM
        return bytes(self._buf)


# ---------------------------------------------------------------------------
# Utility – wrap raw PCM in a WAV container so browsers can play it easily.
# ---------------------------------------------------------------------------
_DEF_SR = 24_000  # ElevenLabs streaming output sample-rate

def _pcm_to_wav(pcm: bytes, *, sample_rate: int = _DEF_SR) -> bytes:
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    buf.flush()
    return Path(buf.name).read_bytes()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class SpeechResponse(BaseModel):
    transcript: str
    tts_audio_b64: str  # base64-encoded WAV (mono 24-kHz)

# ---------------------------------------------------------------------------
# Helper – run SpeechClient pipeline and return transcript + wav bytes
# ---------------------------------------------------------------------------
async def _run_pipeline(source):  # type: ignore[ANN001]
    """Utility shared by HTTP & WebSocket handlers."""
    ai_client = AIClient(providers=[OpenAIProvider()])
    stt_adapter = FireworksSTT()
    tts_adapter = ElevenLabsTTS()
    sink = _BufferSink()

    transcript_parts: List[str] = []

    def _on_user_stop(payload):  # noqa: D401, ANN001
        if "partial" in payload:
            transcript_parts.append(payload["partial"])

    async with ai_client, SpeechClient(ai_client=ai_client,
                                       source=source,
                                       stt=stt_adapter,
                                       tts=tts_adapter,
                                       sink=sink) as sc:
        sc.on("user_stop", _on_user_stop)
        await sc.converse()

    wav_reply = _pcm_to_wav(pcm=sink.pcm)
    return " ".join(transcript_parts).strip(), wav_reply


# ---------------------------------------------------------------------------
# Main endpoint – one user utterance → one assistant reply
# ---------------------------------------------------------------------------
@app.post("/api/speech", response_model=SpeechResponse)
async def speech_endpoint(file: UploadFile = File(...)) -> SpeechResponse:  # noqa: D401
    """Accept an audio file, run the SpeechClient pipeline and reply with text+audio."""

    # ------------------------------------------------------------------ persist upload to tmp file
    raw_suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    raw_path = Path(tempfile.mktemp(suffix=raw_suffix))
    raw_path.write_bytes(await file.read())

    # ------------------------------------------------------------------ ensure WAV format for FileSource
    user_wav_path = raw_path  # optimistic – may already be wav
    try:
        # Quick check – try opening with wave (RIFF). Will raise if not WAV.
        with wave.open(str(user_wav_path), "rb") as _:
            pass  # success ⇒ already WAV
    except wave.Error:
        # Need conversion via ffmpeg ----------------------------------------------------
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Uploaded audio not WAV and ffmpeg not found for conversion")

        conv_path = raw_path.with_name(raw_path.stem + "_conv.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-ac", "1",           # mono
            "-ar", "16000",       # 16-kHz sample-rate (STT friendly)
            "-vn",                # no video
            str(conv_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not conv_path.exists():
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode('utf-8', 'ignore')}")
        user_wav_path = conv_path

    # Instantiate adapters --------------------------------------------------
    ai_client = AIClient(providers=[OpenAIProvider()])
    stt_adapter = FireworksSTT()  # Switch to WhisperSTT if you lack an API key
    tts_adapter = ElevenLabsTTS()
    sink = _BufferSink()
    source = FileSource(user_wav_path)

    transcript_parts: List[str] = []

    # Small helper to capture final user transcript from pipeline events
    def _on_user_stop(payload):  # noqa: D401, ANN001 – non-async callback is fine
        if "partial" in payload:
            transcript_parts.append(payload["partial"])

    # Run the speech pipeline inside a single async context ---------------
    async with ai_client, SpeechClient(ai_client=ai_client,
                                       source=source,
                                       stt=stt_adapter,
                                       tts=tts_adapter,
                                       sink=sink) as sc:
        sc.on("user_stop", _on_user_stop)
        await sc.converse()  # blocks until FileSource is exhausted

    full_transcript = " ".join(transcript_parts).strip()

    wav_reply = _pcm_to_wav(pcm=sink.pcm)
    b64_audio = base64.b64encode(wav_reply).decode("ascii")

    return SpeechResponse(transcript=full_transcript, tts_audio_b64=b64_audio)

# ---------------------------------------------------------------------------
# WebSocket endpoint for streaming audio (16-kHz mono PCM chunks)
# ---------------------------------------------------------------------------

# Minimal AudioSource that consumes bytes from an asyncio.Queue populated by WS
from typing import AsyncIterator


@app.websocket("/ws/speech")
async def ws_speech(ws: WebSocket) -> None:  # noqa: D401
    """WebSocket endpoint.

    Client must send raw PCM 16-bit little-endian 16-kHz mono chunks.
    Send the literal string "__end__" (text) or a zero-length binary frame to
    mark end-of-utterance.  Server responds with a JSON message:
        {"transcript": "...", "tts_audio_b64": "..."}
    before closing.
    """

    await ws.accept()

    # ------------------------------------------------------------------ receive full recording
    buf_parts: list[bytes] = []

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.receive":
                if "bytes" in msg and msg["bytes"] is not None and len(msg["bytes"]):
                    buf_parts.append(msg["bytes"])
                else:
                    # zero-length frame → end
                    break
                if "text" in msg and msg["text"] == "__end__":
                    break
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass

    raw_bytes = b"".join(buf_parts)

    # Persist to temp file and reuse HTTP conversion + pipeline logic -------
    tmp_path = Path(tempfile.mktemp(suffix=".bin"))
    tmp_path.write_bytes(raw_bytes)

    # Reuse conversion logic from earlier handler --------------------------
    def _convert_bin_to_wav(path: Path) -> Path:  # noqa: D401
        try:
            with wave.open(str(path), "rb") as _:
                return path  # already WAV
        except wave.Error:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg required for non-WAV streaming upload")
            out_path = path.with_name(path.stem + "_conv.wav")
            cmd = ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", "-vn", str(out_path)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if not out_path.exists():
                raise RuntimeError("ffmpeg conversion failed")
            return out_path

    wav_path = _convert_bin_to_wav(tmp_path)

    transcript, wav_reply = await _run_pipeline(FileSource(wav_path))

    await ws.send_json({"transcript": transcript,
                        "tts_audio_b64": base64.b64encode(wav_reply).decode("ascii")})
    await ws.close() 