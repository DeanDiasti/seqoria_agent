"""ElevenLabs streaming Text-to-Speech adapter.

This adapter uses the **websocket streaming** endpoint to obtain low-latency
PCM audio suitable for real-time playback.  It intentionally keeps all
networking imports *lazy* so that importing the module will not error when the
`elevenlabs` or `websockets` extras are missing - instantiation raises instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
import httpx
import websockets
from typing import AsyncIterator, Set

import base64
import json
import urllib.parse


from seqoria_agent.models.speech_models import AudioChunk
from seqoria_agent.tts_providers.base import BaseTTSAdapter

__all__: list[str] = ["ElevenLabsTTS"]

_LOG = logging.getLogger(name=__name__)


class ElevenLabsTTS(BaseTTSAdapter):
    """Stream TTS audio from the ElevenLabs websocket API."""

    _voice_cache: Set[str] = set()
    _cache_lock = asyncio.Lock()

    def __init__(self, 
                 voice: str = "Xb7hH8MSUJpSbSDYk0k2", 
                 *, 
                 model: str = "eleven_flash_v2_5",
                 api_key: str | None = None,
                 # ElevenLabs voice setting overrides -----------------------------------
                 stability: float | None = 0.85,
                 similarity_boost: float | None = 0.85,
                 style: float | None = None,
                 speed: float | None = None,
                 use_speaker_boost: bool | None = None,
                 output_format: str = "pcm_24000",
                 batch_words: int | None = None) -> None:  # noqa: D401


        self._voice = voice
        self._model = model
        self._output_format = output_format
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise ValueError("Missing ElevenLabs API key - set ELEVENLABS_API_KEY or pass api_key")

        # Aggregate provided voice settings – ignore ``None`` values to fall
        # back to ElevenLabs defaults when a parameter is not explicitly
        # overridden.

        self._voice_settings = {
            k: v for k, v in {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "speed": speed,
                "use_speaker_boost": use_speaker_boost,
            }.items() if v is not None
        }

        # Store optional batching threshold (None or <=0 disables batching)
        self._batch_words: int | None = int(batch_words) if batch_words else None

        # Validate streaming compatibility for chosen output format.
        if not self._output_format.startswith("pcm_"):
            _LOG.debug("ElevenLabsTTS: Non-PCM output format '%s' selected - will work for generate() but not stream().", self._output_format)

        # Kick-off voice list fetch in background so init stays fast.
        asyncio.create_task(self._ensure_voice_cache())

    # BaseTTSAdapter interface

    async def stream(self,
                     text_iter: AsyncIterator[str],
                     *,
                     voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Stream *text_iter* via the ElevenLabs **multi-context WebSocket** API.

        This adapter no longer falls back to the blocking SDK helper – a valid
        `websockets` installation is therefore required at runtime.
        """

        voice_id = voice or self._voice

        # Wrap input iterator with batching if enabled
        async def _batched(inp: AsyncIterator[str]):  # noqa: D401
            if not (self._batch_words and self._batch_words > 0):
                async for txt in inp:
                    yield txt
                return

            buf: str = ""
            async for txt in inp:
                if not txt:
                    continue
                # Ensure single space separation
                buf = (buf + " " + txt).strip() if buf else txt
                if len(buf.split()) >= self._batch_words:
                    # ElevenLabs requires trailing space for WS proto
                    if not buf.endswith(" "):
                        buf += " "
                    yield buf
                    buf = ""
            if buf:
                if not buf.endswith(" "):
                    buf += " "
                yield buf

        text_iter = _batched(text_iter)

        async for chunk in self._stream_ws(text_iter=text_iter, voice_id=voice_id):
            yield chunk

    # Preferred WebSocket backend – incremental, low-latency streaming
    async def _stream_ws(self, *, text_iter: AsyncIterator[str], voice_id: str) -> AsyncIterator[AudioChunk]:  # noqa: D401
        """Stream using the *multi-context WebSocket* endpoint."""

        # Validate output format – WS currently only usable for raw PCM.
        if not self._output_format.startswith("pcm_"):
            raise RuntimeError("WebSocket streaming requires a 'pcm_<rate>' output_format, got " + self._output_format)

        # Build endpoint URL with query string.
        params = {"model_id": self._model,
                  "output_format": self._output_format}
        qs = urllib.parse.urlencode(query=params)
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input"+ (f"?{qs}" if qs else "")

        headers = {"xi-api-key": self._api_key}

        context_id = "default"

        # Derive sample rate once from output_format (pcm_<rate>).
        try:
            sample_rate = int(self._output_format.split(sep="_")[1])
        except Exception:
            sample_rate = 24000

        async with websockets.connect(uri=url, additional_headers=headers) as ws:
            # 1️⃣  initialiseConnectionMulti – a single space kick-starts the context
            init_msg = {"text": " ", "context_id": context_id}
            if self._voice_settings:
                init_msg["voice_settings"] = self._voice_settings
            await ws.send(message=json.dumps(obj=init_msg))

            # Launch sender task that proxies *text_iter* → WS messages.
            async def _sender() -> None:  # noqa: D401
                async for chunk in text_iter:
                    if not chunk.endswith(" "):
                        chunk += " "  # protocol requires trailing space
                    msg = {"text": chunk,
                           "context_id": context_id}
                    await ws.send(message=json.dumps(obj=msg))

                # Final flush & graceful shutdown of the context.
                await ws.send(message=json.dumps(obj={"context_id": context_id, "flush": True}))
                await ws.send(message=json.dumps(obj={"context_id": context_id, "close_context": True}))

                # Close the underlying socket once all audio flushed.
                await ws.send(message=json.dumps(obj={"close_socket": True}))

            sender_task = asyncio.create_task(coro=_sender(), 
                                              name="ElevenLabsTTS-WS-Sender")

            try:
                async for raw in ws:
                    data = json.loads(raw)

                    # Handle audioOutputMulti messages
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        try:
                            pcm_bytes = base64.b64decode(audio_b64)
                            yield AudioChunk(pcm=pcm_bytes, sample_rate=sample_rate, channels=1)
                        except Exception:
                            _LOG.debug("Failed to decode ElevenLabs audio chunk", exc_info=True)

                    # Stop when we receive the *final* sentinel for our context.
                    if data.get("is_final") or data.get("isFinal"):
                        break
            finally:
                sender_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sender_task

    async def generate(self,
                       text: str,
                       *,
                       voice: str | None = None) -> bytes:
        """Blocking-style helper that returns the full PCM buffer for *text*.

        Internally this reuses the WebSocket transport and simply aggregates
        all streamed chunks into memory.  Suitable for short prompts or
        unit-tests where a single reply is required.
        """

        if not text:
            return b""

        voice_id = voice or self._voice

        # Build JSON payload in line with the Create Speech endpoint.
        payload = {"text": text,
                   "model_id": self._model,
                   "output_format": self._output_format}

        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url=url, 
                                     json=payload, 
                                     headers={"xi-api-key": self._api_key})
            if resp.status_code != 200:
                raise RuntimeError(f"ElevenLabsTTS.generate failed: {resp.status_code} - {resp.text[:200]}")

            return resp.content

    # Capability helpers ------------------------------------------------

    @classmethod
    async def _ensure_voice_cache(cls) -> None:  # noqa: D401
        if cls._voice_cache:
            return  # already populated
        async with cls._cache_lock:  # prevent concurrent fetches
            if cls._voice_cache:
                return
            try:  # type: ignore[attr-defined]

                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get("https://api.elevenlabs.io/v1/voices")
                    if resp.status_code == 200:
                        voices = resp.json().get("voices", [])
                        cls._voice_cache.update({v.get("voice_id") for v in voices if v.get("voice_id")})
                        cls._voice_cache.update({v.get("name") for v in voices if v.get("name")})
            except Exception as exc:  # pragma: no cover – network errors ignored
                _LOG.debug("Could not refresh ElevenLabs voice cache: %s", exc)

    # ------------------------------------------------------------------
    @classmethod
    def supports_voice(cls, voice: str) -> bool:  # noqa: D401
        return not cls._voice_cache or voice in cls._voice_cache 