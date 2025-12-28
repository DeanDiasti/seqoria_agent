from __future__ import annotations

# Standard library
import io
import logging
# wave module no longer needed – we'll craft header manually
import asyncio
import contextlib
import os
import struct
import time
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import string as _str

# Third-party
import httpx
import ffmpeg

from typing import AsyncIterator, List, Optional, Dict, Tuple, Callable, Awaitable

from seqoria_agent.models.speech_models import AudioChunk, TranscriptChunk
from seqoria_agent.stt_providers.base import BaseSTTAdapter

__all__: list[str] = ["FireworksSTT"]

# ---------------------------------------------------------------------------
# Normalisation helpers – fast punctuation/whitespace stripping
# ---------------------------------------------------------------------------

# Build a translate table that removes punctuation & whitespace characters.
# `None` in the mapping deletes the character.
_TRANSLATE_TABLE: dict[int, None] = {
    ord(ch): None for ch in (_str.punctuation + _str.whitespace)
}

# Fast membership test for detecting relevant characters (non-whitespace, non-punctuation).
_CONTENT_CHARS: set[str] = {
    chr(c) for c in range(32, 127)
    if chr(c) not in _str.whitespace and chr(c) not in _str.punctuation
}


def _normalise_text(s: str) -> str:  # noqa: D401 – helper
    """Return lowercase *s* with punctuation/space removed (fast)."""

    # Using translate → lower is significantly faster than per-char loops.
    return s.translate(_TRANSLATE_TABLE).lower()


def _split_new_text(prev_norm: str, text: str) -> tuple[str, str]:  # noqa: D401 helper
    """Return (new_text, new_prev_norm).

    Removes any leading overlap with *prev_norm* according to normalised
    matching.  Also returns the updated prev_norm (trimmed to keep history
    small by caller).
    """

    norm_text = _normalise_text(text)

    # Fast path: nothing to overlap
    if not prev_norm:
        return text, norm_text

    max_overlap = min(len(prev_norm), len(norm_text))
    overlap_len = 0
    for l in range(max_overlap, 0, -1):
        if prev_norm[-l:] == norm_text[:l]:
            overlap_len = l
            break

    if overlap_len == len(norm_text):
        # Completely duplicate
        return "", prev_norm

    if not overlap_len:
        return text, prev_norm + norm_text

    # Determine cut index in *original* text where overlap ends.
    norm_counter = 0
    cut_idx = 0
    for idx, ch in enumerate(text):
        if ch in _CONTENT_CHARS:
            norm_counter += 1
        if norm_counter >= overlap_len:
            cut_idx = idx + 1
            break
    new_text = text[cut_idx:].lstrip()
    return new_text, prev_norm + norm_text[overlap_len:]


_LOG = logging.getLogger(name=__name__)

DEFAULT_SAMPLE_RATE = 16_000  # Hz
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_SEC = 0.5  # seconds

# ---------------------------------------------------------------------------
# Public helper type aliases (for external type checkers convenience)
# ---------------------------------------------------------------------------
SourceUrl = str  # URL to remote media (live stream or file)
SourcePath = os.PathLike[str] | str  # local media path


class FireworksSTT(BaseSTTAdapter):
    """Speech-to-text adapter using Fireworks Whisper endpoints.

    Because Fireworks currently offers *batch* transcription only, this adapter
    fakes real-time streaming by buffering roughly ``chunk_duration`` milliseconds of
    audio (with ``overlap`` milliseconds of history) before sending it to the remote
    service.  The returned segments are timestamp-adjusted and yielded as
    incremental :class:`~seqoria_agent.models.speech_models.TranscriptChunk`
    objects.

    Parameters
    ----------
    api_key:
        Your *FIREWORKS_API_KEY* value.  When *None*, the ``FIREWORKS_API_KEY``
        environment variable is consulted.
    model:
        Name of the Fireworks ASR model to use.  Either ``"whisper-v3"`` or
        ``"whisper-v3-turbo"``.
    base_url:
        Endpoint prefix *without* the ``/v1`` suffix.  Defaults to the prod
        endpoint for ``whisper-v3``.
    chunk_duration:
        Target duration (in milliseconds) for each remote request.
    overlap:
        Amount of audio (in milliseconds) kept from the previous chunk to improve
        continuity.  Must be smaller than *chunk_duration*.
    """

    def __init__(self,
                 *,
                 api_key: str | None = None,
                 model: str = "whisper-v3-turbo",
                 base_url: str | None = None,
                 chunk_duration: float = 5000.0,
                 overlap: float = 1000.0,
                 additional_prompt: str | None = None,
                 max_inflight: int = 2,
                 metrics_cb: Optional[Callable[[str, float], Awaitable[None]]] = None) -> None:

        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        if not (0 <= overlap < chunk_duration):
            raise ValueError("overlap must be >=0 and < chunk_duration")

        self._api_key: str | None = api_key or os.getenv(key="FIREWORKS_API_KEY")
        if not self._api_key:
            raise ValueError("FIREWORKS_API_KEY not provided")

        self._model = model
        self._base_url = (base_url or "https://audio-turbo.us-virginia-1.direct.fireworks.ai")

        # Store developer-supplied extra prompt (may be empty)
        self._additional_prompt = (additional_prompt or "").strip()

        # Convert milliseconds → seconds for internal use
        self._chunk_dur = float(chunk_duration) / 1000.0
        self._overlap = float(overlap) / 1000.0

        # HTTP client (single instance, HTTP/2 enabled)
        self._client: "httpx.AsyncClient" = httpx.AsyncClient(timeout=30.0, http2=True)

        # HTTP retry behaviour
        self._max_retries: int = 3  # configurable by advanced users
        self._initial_backoff: float = 0.5  # seconds (doubles each retry)

        # Concurrency controls – how many _transcribe requests in flight
        self._sem = asyncio.Semaphore(max_inflight)

        # Metrics callback for latency / bytes etc. Signature: (name, value)
        self._metrics_cb = metrics_cb

        # (Opt-in) send raw 16-bit PCM instead of WAV when True. Fireworks
        # deployments generally expect standard audio containers; leave off by
        # default.
        self._send_pcm_raw: bool = False

        # Cached ffmpeg decode process for streaming reuse.
        self._decode_proc: subprocess.Popen | None = None
        self._decode_proc_src: str | None = None

        # ------------------------------------------------------------------
        # Pre-computed buffer sizes & dedup history limits
        # ------------------------------------------------------------------
        self._bytes_per_sample: int = 2  # 16-bit PCM
        self._chunk_bytes_default: int = int(
            DEFAULT_SAMPLE_RATE * DEFAULT_CHANNELS * self._bytes_per_sample * self._chunk_dur
        )
        self._overlap_bytes_default: int = int(
            DEFAULT_SAMPLE_RATE * DEFAULT_CHANNELS * self._bytes_per_sample * self._overlap
        )

        # Keep generous history to avoid duplicate text artefacts.
        self._max_char_history: int = 2048

    async def _maybe_metric(self, name: str, value: float) -> None:  # noqa: D401
        """Emit *value* for *name* through the metrics callback if supplied."""
        if self._metrics_cb is None:
            return

        try:
            maybe_coro = self._metrics_cb(name, value)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:  # noqa: BLE001 – never raise from metrics path
            _LOG.debug("Metrics callback failed for %s", name, exc_info=True)

    # ------------------------------------------------------------------
    # BaseSTTAdapter interface
    # ------------------------------------------------------------------

    async def stream(self,
                     *,
                     frames: Optional[AsyncIterator[AudioChunk]] = None,
                     url: Optional[SourceUrl] = None,
                     file: Optional[SourcePath] = None,
                     language: str | None = None,
                     frame_sec: float = DEFAULT_FRAME_SEC) -> AsyncIterator[TranscriptChunk]:
        """Incrementally transcribe *one* of *frames*, *url* or *file*.

        Exactly **one** of the three sources must be provided:

        • ``frames`` - an ``AsyncIterator[AudioChunk]`` (legacy behaviour).
        • ``url``    - remote media (live radio or downloadable file).
        • ``file``   - local audio/video file path.
        """

        provided = sum(x is not None for x in (frames, url, file))
        if provided == 0:
            raise ValueError("Must provide one of 'frames', 'url', or 'file'.")
        if provided > 1:
            raise ValueError("Only one of 'frames', 'url', or 'file' can be supplied at a time.")

        # ------------------------------------------------------------------
        # Normalise into an AsyncIterator[AudioChunk]
        # ------------------------------------------------------------------
        if frames is not None:
            frames_iter: AsyncIterator[AudioChunk] = frames
        elif url is not None:
            frames_iter = self._frames_from_source(src=url, frame_sec=frame_sec)
        else:  # file is not None by validation above
            frames_iter = self._frames_from_source(src=os.fspath(file), frame_sec=frame_sec)

        sample_rate: int | None = None
        channels: int | None = None
        buf = bytearray()

        # Compute byte counts lazily once we know the sample_rate/channels.
        chunk_bytes: int | None = None
        overlap_bytes: int | None = None

        # Running timestamp of *first* byte in the buffer
        base_ts: float = 0.0

        # ------------------------------------------------------------------
        # Deduplication state – keeps a rolling window of the tokens that were
        # already yielded to the caller so we can remove duplicates stemming
        # from the overlap between consecutive chunks.
        # ------------------------------------------------------------------

        prev_norm: str = ""  # trailing normalised chars we already emitted

        # Rolling context text that will be fed back as "prompt" to Fireworks.
        context_text: str = ""
        CONTEXT_CHAR_LIMIT = 400  # keep prompt size reasonable

        seq_idx: int = 0  # order of launched chunks
        tasks: Dict[int, asyncio.Task[Tuple[int, List[TranscriptChunk]]]] = {}
        next_yield_idx: int = 0

        async for chunk in frames_iter:
            if sample_rate is None:
                # Initialisation once first frame arrives
                sample_rate = chunk.sample_rate
                channels = chunk.channels

                if sample_rate == DEFAULT_SAMPLE_RATE and channels == DEFAULT_CHANNELS:
                    chunk_bytes = self._chunk_bytes_default
                    overlap_bytes = self._overlap_bytes_default
                else:
                    # Fallback – compute once for this stream
                    chunk_bytes = int(sample_rate * channels * self._bytes_per_sample * self._chunk_dur)
                    overlap_bytes = int(sample_rate * channels * self._bytes_per_sample * self._overlap)
                _LOG.debug("FireworksSTT configured with chunk_bytes=%d, overlap_bytes=%d", chunk_bytes, overlap_bytes)

            buf.extend(chunk.pcm)

            # If enough audio accumulated, perform transcription
            if chunk_bytes and len(buf) >= chunk_bytes:
                current_audio = bytes(buf[:chunk_bytes])

                # Keep overlap for context of next chunk
                buf = bytearray(buf[chunk_bytes - overlap_bytes :])  # type: ignore[slice-index]

                # Build prompt for this chunk
                prompt_parts: list[str] = []
                if context_text:
                    prompt_parts.append(context_text)
                if self._additional_prompt:
                    prompt_parts.append(self._additional_prompt)
                prompt_str = " ".join(prompt_parts) if prompt_parts else None

                # --------------------------------------------------
                # Launch transcription concurrently
                # --------------------------------------------------

                async def _bounded(seq: int, pcm: bytes, offset_val: float, prompt_val: Optional[str]):  # noqa: D401
                    async with self._sem:
                        t_start = time.monotonic()
                        res = await self._transcribe(pcm=pcm,
                                                     sample_rate=sample_rate,  # type: ignore[arg-type]
                                                     channels=channels,  # type: ignore[arg-type]
                                                     language=language,
                                                     offset=offset_val,
                                                     final=False,
                                                     prompt=prompt_val)
                        latency = time.monotonic() - t_start
                        await self._maybe_metric("fireworks.latency", latency)
                        await self._maybe_metric("fireworks.bytes_sent", len(pcm))
                        return seq, res

                tasks[seq_idx] = asyncio.create_task(
                    _bounded(seq_idx, current_audio, base_ts, prompt_str)
                )

                seq_idx += 1

                # Drain ready tasks in order
                while True:
                    task = tasks.get(next_yield_idx)
                    if task is None or not task.done():
                        break
                    done_seq, transcripts_res = await tasks.pop(next_yield_idx)

                    for t in transcripts_res:
                        # --------------------------------------------------
                        # Remove already-emitted prefixes (dedup logic)
                        # --------------------------------------------------
                        text = t.text.strip()
                        if not text:
                            continue

                        new_text, prev_norm = _split_new_text(prev_norm, text)
                        if not new_text:
                            continue

                        # Trim history to reasonable length.
                        if len(prev_norm) > self._max_char_history:
                            prev_norm = prev_norm[-self._max_char_history:]

                        yield TranscriptChunk(text=new_text,
                                              is_final=t.is_final,
                                              start_ts=t.start_ts,
                                              end_ts=t.end_ts,
                                              role=t.role)

                        context_text = (context_text + " " + new_text).strip()
                        if len(context_text) > CONTEXT_CHAR_LIMIT:
                            context_text = context_text[-CONTEXT_CHAR_LIMIT:]

                    next_yield_idx += 1

                # Advance base_ts – we progressed chunk_duration-overlap seconds
                base_ts += self._chunk_dur - self._overlap

        # ------------------------------------------------------------------
        # Stream exhausted – await any remaining transcription tasks
        # ------------------------------------------------------------------
        while tasks:
            task = tasks.get(next_yield_idx)
            if task is None:
                # Wait briefly until the expected task index completes/is created
                await asyncio.sleep(0.05)
                continue

            done_seq, transcripts_res = await tasks.pop(next_yield_idx)

            for t in transcripts_res:
                text = t.text.strip()
                if not text:
                    continue

                new_text, prev_norm = _split_new_text(prev_norm, text)
                if not new_text:
                    continue

                if len(prev_norm) > self._max_char_history:
                    prev_norm = prev_norm[-self._max_char_history:]

                yield TranscriptChunk(text=new_text,
                                      is_final=t.is_final,
                                      start_ts=t.start_ts,
                                      end_ts=t.end_ts,
                                      role=t.role)

                context_text = (context_text + " " + new_text).strip()
                if len(context_text) > CONTEXT_CHAR_LIMIT:
                    context_text = context_text[-CONTEXT_CHAR_LIMIT:]

        # ------------------------------------------------------------------
        # Flush remaining buffered audio (< chunk_bytes) at stream end
        # ------------------------------------------------------------------
        if buf and sample_rate is not None and channels is not None:
            prompt_parts: list[str] = []
            if context_text:
                prompt_parts.append(context_text)
            if self._additional_prompt:
                prompt_parts.append(self._additional_prompt)
            prompt_str = " ".join(prompt_parts) if prompt_parts else None

            transcripts = await self._transcribe(pcm=bytes(buf),
                                                 sample_rate=sample_rate,
                                                 channels=channels,
                                                 language=language,
                                                 offset=base_ts,
                                                 final=True,
                                                 prompt=prompt_str)

            for t in transcripts:
                text = t.text.strip()
                if not text:
                    continue

                new_text, prev_norm = _split_new_text(prev_norm, text)
                if not new_text:
                    continue

                if len(prev_norm) > self._max_char_history:
                    prev_norm = prev_norm[-self._max_char_history:]

                yield TranscriptChunk(text=new_text,
                                      is_final=t.is_final,
                                      start_ts=t.start_ts,
                                      end_ts=t.end_ts,
                                      role=t.role)
                context_text = (context_text + " " + new_text).strip()
                if len(context_text) > CONTEXT_CHAR_LIMIT:
                    context_text = context_text[-CONTEXT_CHAR_LIMIT:]


    async def transcribe(self,
                         *,
                         pcm: bytes | None = None,
                         url: Optional[SourceUrl] = None,
                         file: Optional[SourcePath] = None,
                         sample_rate: int = DEFAULT_SAMPLE_RATE,
                         channels: int = DEFAULT_CHANNELS,
                         language: str | None = None) -> List[TranscriptChunk]:
        """Transcribe a complete buffer or media source.

        Exactly **one** of ``pcm``, ``url`` or ``file`` must be provided.
        If ``pcm`` is given it *must* already be 16-bit little-endian PCM at
        *sample_rate* and *channels*.
        """

        # Early out if requested language unsupported.
        if language and not self.supports_language(language):
            _LOG.warning("Requested language '%s' not supported by FireworksSTT", language)
            return []

        provided = sum(x is not None for x in (pcm, url, file))
        if provided == 0:
            raise ValueError("Must provide one of 'pcm', 'url', or 'file'.")
        if provided > 1:
            raise ValueError("Only one of 'pcm', 'url', or 'file' can be supplied at a time.")

        if pcm is not None:
            pcm_bytes: bytes = bytes(pcm)
        elif url is not None:
            pcm_bytes = await self._decode_source_to_pcm(src=url, sample_rate=sample_rate, channels=channels)
        else:  # file is not None by validation
            pcm_bytes = await self._decode_source_to_pcm(src=os.fspath(file), sample_rate=sample_rate, channels=channels)

        prompt_str = self._additional_prompt if self._additional_prompt else None
        return await self._transcribe(pcm=pcm_bytes,
                                      sample_rate=sample_rate,
                                      channels=channels,
                                      language=language,
                                      offset=0.0,
                                      final=True,
                                      prompt=prompt_str)

    async def _transcribe(self,
                          pcm: bytes,
                          sample_rate: int,
                          channels: int,
                          language: str | None,
                          *,
                          offset: float,
                          final: bool = False,
                          prompt: str | None = None) -> List[TranscriptChunk]:  # noqa: D401 – helper
        """Send *pcm* bytes to Fireworks and return transcript chunks."""

        # Decide whether to wrap PCM into a WAV container or send raw bytes.
        if self._send_pcm_raw:
            file_tuple = ("audio.raw", pcm, "application/octet-stream")
        else:
            wav_bytes = self._pcm_to_wav(pcm=pcm, sample_rate=sample_rate, channels=channels)
            file_tuple = ("audio.wav", wav_bytes, "audio/wav")

        # HTTP client already initialised in __init__
        url: str = f"{self._base_url}/v1/audio/transcriptions"
        headers = {"Authorization": self._api_key}

        data: dict[str, object] = {"model": self._model,
                                   "response_format": "verbose_json"}
        if prompt:
            data["prompt"] = prompt
        if language:
            data["language"] = language

        files = {"file": file_tuple}

        # ------------------------------------------------------------------
        # Robust HTTP call with simple exponential backoff on 429 / 5xx.
        # ------------------------------------------------------------------
        payload: dict[str, object] | None = None
        for attempt in range(1, self._max_retries + 2):  # retries + initial try
            try:
                resp = await self._client.post(url=url,
                                               headers=headers,
                                               data=data,
                                               files=files)
                # If server still busy, treat as retryable.
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise httpx.HTTPStatusError("Retryable status", request=resp.request, response=resp)

                resp.raise_for_status()
                payload = resp.json()
                break  # success
            except (httpx.HTTPError, httpx.TransportError) as exc:
                if attempt > self._max_retries:
                    _LOG.exception("Fireworks transcription failed after %d attempts: %s", attempt - 1, exc)
                    return []
                backoff = self._initial_backoff * (2 ** (attempt - 1))
                _LOG.warning("FireworksSTT retrying in %.1fs after error: %s", backoff, exc)
                await asyncio.sleep(backoff)
                continue

        if payload is None:
            # Shouldn't happen but guard
            _LOG.error("Fireworks transcription failed - no payload received")
            return []

        # `payload` is already a JSON dictionary.
        segments = payload.get("segments") or []
        results: list[TranscriptChunk] = []
        for seg in segments:
            try:
                text = seg.get("text", "")
                start = float(seg.get("start", 0)) + offset
                end = float(seg.get("end", 0)) + offset
                results.append(TranscriptChunk(text=text,
                                               is_final=final,
                                               start_ts=start,
                                               end_ts=end))
            except Exception as exc:  # noqa: BLE001 – skip bad segment
                _LOG.debug("Skipping segment due to error: %s", exc)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:  # noqa: D401
        """Return WAV bytes for *pcm* without using wave.open (fast)."""

        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        subchunk2_size = len(pcm)
        riff_chunk_size = 36 + subchunk2_size

        # Little-endian (<). Format refs: https://tinyurl.com/wav-spec
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            riff_chunk_size,
            b"WAVE",
            b"fmt ",
            16,  # SubChunk1Size (PCM)
            1,  # AudioFormat = PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            subchunk2_size,
        )
        return header + pcm

    # ------------------------------------------------------------------
    @classmethod
    def supports_language(cls, lang: str) -> bool:
        # Map of two-letter codes supported by Fireworks.  Keep short for now.
        supported = {"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
                     "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                     "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                     "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                     "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                     "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                     "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                     "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                     "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                     "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"}
        return lang in supported

    # ------------------------------------------------------------------
    # Convenience helpers ------------------------------------------------
    # ------------------------------------------------------------------

    async def _frames_from_source(self, src: str, *, frame_sec: float) -> AsyncIterator[AudioChunk]:  # noqa: D401
        """Decode *src* into PCM frames using ffmpeg.

        Supports both local file paths and URLs.  The output is a stream of
        :class:`AudioChunk` instances with the default sample-rate/channels
        expected by the Fireworks Whisper models (16-kHz mono).
        """

        # Determine if *src* is a URL by checking scheme.
        is_url = bool(urlparse(src).scheme and ":" in src)
        input_arg = src if is_url else os.fspath(Path(src).expanduser())

        # ------------------------------------------------------------------
        # Re-use a cached ffmpeg process if streaming the *same* source and it
        # is still alive.  Otherwise spawn a new one and remember it.
        # ------------------------------------------------------------------
        if self._decode_proc_src == src and self._decode_proc and self._decode_proc.poll() is None:
            process = self._decode_proc
        else:
            # Build ffmpeg command via `ffmpeg-python` and launch asynchronously.
            process = (
                ffmpeg
                .input(input_arg)
                .output(
                    "pipe:",
                    format="s16le",
                    ac=DEFAULT_CHANNELS,
                    ar=DEFAULT_SAMPLE_RATE,
                    loglevel="quiet",
                )
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            # Cache for potential reuse.
            self._decode_proc = process
            self._decode_proc_src = src

        bytes_per_frame = int(DEFAULT_SAMPLE_RATE * DEFAULT_CHANNELS * 2 * frame_sec)

        try:
            while True:
                # Use thread off-loading to avoid blocking the event-loop.
                data = await asyncio.to_thread(process.stdout.read, bytes_per_frame)
                if not data:
                    break
                yield AudioChunk(pcm=data, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS)
        finally:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            process.wait()

            if process.returncode not in (0, None):
                try:
                    stderr_data = process.stderr.read().decode(errors="ignore")  # type: ignore[union-attr]
                    _LOG.error("ffmpeg exited with code %s for src %s. Stderr: %s", process.returncode, src, stderr_data)
                except Exception:  # noqa: BLE001
                    _LOG.error("ffmpeg process failed with code %s (stderr unavailable)", process.returncode)

            # If this was the cached process, clear cache on termination.
            if process is self._decode_proc:
                self._decode_proc = None
                self._decode_proc_src = None

    async def _decode_source_to_pcm(self, src: str, *, sample_rate: int, channels: int) -> bytes:  # noqa: D401
        """Return full PCM byte buffer for *src* using ffmpeg."""

        is_url = bool(urlparse(src).scheme and ":" in src)
        input_arg = src if is_url else os.fspath(Path(src).expanduser())

        process = (
            ffmpeg
            .input(input_arg)
            .output(
                "pipe:",
                format="s16le",
                ac=channels,
                ar=sample_rate,
                loglevel="quiet",
            )
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Read the entire pcm buffer in a worker thread (blocking IO).
        stdout_data, _ = await asyncio.to_thread(process.communicate)
        return stdout_data

    async def aclose(self) -> None:  # noqa: D401 – public resource cleanup
        """Release network sessions and kill any cached ffmpeg process."""

        if self._client is not None:
            await self._client.aclose()

        if self._decode_proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._decode_proc.kill()
            self._decode_proc.wait()
            self._decode_proc = None
            self._decode_proc_src = None

    # --------------------------------------------------------------
    # Async context-manager helpers
    # --------------------------------------------------------------

    async def __aenter__(self):  # noqa: D401 – context manager helper
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        await self.aclose()
