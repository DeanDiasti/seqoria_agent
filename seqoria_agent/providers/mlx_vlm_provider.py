"""
Local vision-language model provider using MLX-VLM (mlx_vlm) on Apple Silicon.

Loads VLMs from a Hugging Face repo id or local path; uses ``apply_chat_template`` and
``mlx_vlm.generate``. Blocking inference runs in ``asyncio.to_thread`` for async providers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import builtins
import importlib.util
import os
import platform
import tempfile
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

from seqoria_agent.client.ai_client.utils import json_dumps, json_loads
from seqoria_agent.models.chunk import (
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolCallResult,
    ToolCallError,
    UserMessage,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider
from seqoria_agent.providers.errors import (
    ConnectionError,
    ContextLengthExceededError,
    EnvironmentError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    ServerError,
)
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage

_MLX_VLM_GEN_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "repetition_context_size",
        "verbose",
        "seed",
    }
)


def mlx_vlm_available() -> bool:
    return importlib.util.find_spec("mlx_vlm") is not None


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise EnvironmentError(
            message="MLX-VLM requires macOS on Apple Silicon (arm64).",
            missing_vars=[],
        )


def _import_mlx_vlm() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.generate import stream_generate
    except ImportError as exc:
        raise EnvironmentError(
            message="mlx_vlm is not installed. Install with: pip install 'seqoria-agent[mlx-vlm]'",
            missing_vars=[],
        ) from exc
    try:
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise EnvironmentError(
            message="mlx_vlm.prompt_utils is missing; upgrade mlx-vlm.",
            missing_vars=[],
        ) from exc
    try:
        from mlx_vlm.utils import load_config
    except ImportError:
        load_config = None  # type: ignore[assignment]
    return load, generate, stream_generate, apply_chat_template, load_config


class _MLXVLMProviderMixin:
    model_id: str
    max_retries: int
    default_params: Dict[str, Any]
    _model: Any
    _processor: Any
    _config: Any

    def _init_common(self, model_id: str, **kwargs: Any) -> Tuple[str, Dict[str, Any], int]:
        if not model_id or not str(model_id).strip():
            raise InvalidRequestError(message="model_id is required for MLX-VLM provider")
        max_retries = int(kwargs.get("max_retries", 3))
        default_params: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _MLX_VLM_GEN_KEYS and key != "max_retries":
                default_params[key] = value
        return str(model_id), default_params, max_retries

    def _prepare_params(self, **kwargs: Any) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update(kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def _mlx_vlm_generate_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in params.items() if k in _MLX_VLM_GEN_KEYS}

    def _ensure_loaded(self) -> None:
        if getattr(self, "_model", None) is not None and getattr(self, "_processor", None) is not None:
            return
        load_fn, _, _, _, load_config_fn = _import_mlx_vlm()
        try:
            self._model, self._processor = load_fn(self.model_id)
            if load_config_fn is not None:
                self._config = load_config_fn(self.model_id)
            else:
                self._config = getattr(self._model, "config", None)
        except Exception as exc:
            raise self._handle_error(exc) from exc

    def _convert_multimodal_content_to_text_and_refs(
        self, content: Union[str, List[Dict[str, Any]]], tmp_dir: str
    ) -> Tuple[str, List[str], List[str]]:
        """Ollama-style multimodal blocks → text plus image and audio paths/URLs for mlx_vlm."""
        if isinstance(content, str):
            return content, [], []

        text_parts: List[str] = []
        images: List[str] = []
        audios: List[str] = []

        for idx, block in enumerate(content):
            btype = block.get("type")

            if btype == "text":
                text_parts.append(str(block.get("text", "")))
                continue

            if btype == "image":
                if "base64" in block:
                    path = self._write_b64_to_temp(block["base64"], tmp_dir, idx, suffix=".png")
                    if path:
                        images.append(path)
                elif "url" in block:
                    ref = self._normalize_media_ref(str(block["url"]), tmp_dir, idx, ".png")
                    if ref:
                        images.append(ref)
                continue

            if btype == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                ref = self._normalize_media_ref(url, tmp_dir, idx, ".png")
                if ref:
                    images.append(ref)
                continue

            if btype == "input_audio":
                ia = block.get("input_audio")
                if isinstance(ia, dict):
                    if "data" in ia:
                        fmt = str(ia.get("format", "wav")).lstrip(".") or "wav"
                        path = self._write_b64_to_temp(
                            str(ia["data"]), tmp_dir, idx, suffix=f".{fmt}"
                        )
                        if path:
                            audios.append(path)
                    elif "url" in ia:
                        ref = self._normalize_media_ref(str(ia["url"]), tmp_dir, idx, ".wav")
                        if ref:
                            audios.append(ref)
                continue

            if btype == "audio":
                if "base64" in block:
                    path = self._write_b64_to_temp(block["base64"], tmp_dir, idx, suffix=".wav")
                    if path:
                        audios.append(path)
                elif "url" in block:
                    ref = self._normalize_media_ref(str(block["url"]), tmp_dir, idx, ".wav")
                    if ref:
                        audios.append(ref)
                continue

            if btype == "audio_url":
                au = block.get("audio_url", {})
                url = au.get("url", "") if isinstance(au, dict) else str(au)
                ref = self._normalize_media_ref(url, tmp_dir, idx, ".wav")
                if ref:
                    audios.append(ref)
                continue

        return " ".join(text_parts).strip(), images, audios

    @staticmethod
    def _write_b64_to_temp(b64: str, tmp_dir: str, idx: int, *, suffix: str = ".png") -> Optional[str]:
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            return None
        path = os.path.join(tmp_dir, f"media_{idx}{suffix}")
        with open(path, "wb") as f:
            f.write(raw)
        return path

    def _normalize_media_ref(self, url: str, tmp_dir: str, idx: int, default_suffix: str) -> Optional[str]:
        if url.startswith("data:"):
            try:
                b64 = url.split(",", 1)[1]
            except IndexError:
                return None
            return self._write_b64_to_temp(b64, tmp_dir, idx, suffix=default_suffix)
        if url.startswith(("http://", "https://")):
            return url
        if os.path.isfile(url):
            return url
        return None

    # Backwards-compatible name for image-only callers
    def _normalize_image_ref(self, url: str, tmp_dir: str, idx: int) -> Optional[str]:
        return self._normalize_media_ref(url, tmp_dir, idx, ".png")

    @staticmethod
    def _text_only_content(content: Union[str, List[Dict[str, Any]]]) -> str:
        """Strip multimodal blocks to text + placeholders (no temp files)."""
        if isinstance(content, str):
            return content
        text_parts: List[str] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype in ("image", "image_url"):
                text_parts.append("[image]")
            elif btype in ("audio", "audio_url", "input_audio"):
                text_parts.append("[audio]")
        return " ".join(text_parts).strip()

    def _append_dict_message(
        self, raw_msg: Dict[str, Any], prepared: List[Dict[str, str]]
    ) -> None:
        role = raw_msg.get("role")
        content = raw_msg.get("content", "")

        if role == "tool_call":
            prepared.append(
                {
                    "role": "assistant",
                    "content": f"[tool_request] {raw_msg.get('name')} {json_dumps(raw_msg.get('arguments', {}))}",
                }
            )
            return

        if role in {"tool_result", "tool_error", "tool"}:
            prepared.append(
                {
                    "role": "user",
                    "content": (f"[tool_error] {raw_msg.get('name', '')} (call_id={raw_msg.get('tool_call_id', '')}): {raw_msg.get('error', '')}"
                                if role == "tool_error"
                                else f"[tool_result] {raw_msg.get('name', '')}: {raw_msg.get('result', content)}"),
                }
            )
            return

        if role in {"assistant", "system", "user"}:
            text_content = self._text_only_content(content) if not isinstance(content, str) else content
            msg: Dict[str, str] = {"role": str(role), "content": text_content}
            tool_calls = raw_msg.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
                parts: List[str] = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json_loads(args)
                            except Exception:
                                args = {}
                        parts.append(f"{fn.get('name', '')} {json_dumps(args if isinstance(args, dict) else {})}")
                if parts:
                    msg["content"] = "[tool_calls] " + " | ".join(parts)
            prepared.append(msg)

    def _prepare_chat_messages_text_only(
        self,
        chat_history: List[
            Union[Dict[str, Any], ToolCallRequest, ToolCallResult, ToolCallError, AssistantMessage, UserMessage, SystemMessage]
        ],
    ) -> List[Dict[str, str]]:
        """String roles and text-only content for chat templates."""
        prepared: List[Dict[str, str]] = []

        for raw_msg in chat_history:
            if isinstance(raw_msg, ToolCallRequest):
                prepared.append(
                    {
                        "role": "assistant",
                        "content": f"[tool_request] {raw_msg.name} {json_dumps(raw_msg.arguments or {})}",
                    }
                )
                continue

            if isinstance(raw_msg, (ToolCallResult, ToolCallError)):
                prepared.append(
                    {
                        "role": "user",
                        "content": (f"[tool_error] {raw_msg.name} (call_id={raw_msg.tool_call_id}): {raw_msg.error}"
                                    if isinstance(raw_msg, ToolCallError)
                                    else f"[tool_result] {raw_msg.name}: {raw_msg.result}"),
                    }
                )
                continue

            if isinstance(raw_msg, (AssistantMessage, UserMessage, SystemMessage)):
                text = (
                    raw_msg.content
                    if isinstance(raw_msg.content, str)
                    else self._text_only_content(raw_msg.content)
                )
                prepared.append({"role": str(raw_msg.role), "content": text})
                continue

            if isinstance(raw_msg, dict):
                self._append_dict_message(raw_msg, prepared)

        return prepared

    def _last_user_message_media(
        self,
        chat_history: List[
            Union[Dict[str, Any], ToolCallRequest, ToolCallResult, ToolCallError, AssistantMessage, UserMessage, SystemMessage]
        ],
        tmp_dir: str,
    ) -> Tuple[List[Dict[str, str]], str, List[str], List[str]]:
        """
        Return (messages, last_user_text, image_refs, audio_refs).

        Images and audio clips are taken only from the **last** user message.
        """
        messages = self._prepare_chat_messages_text_only(chat_history)
        last_user_idx: Optional[int] = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        image_refs: List[str] = []
        audio_refs: List[str] = []
        last_text = ""

        if last_user_idx is None:
            return messages, "", image_refs, audio_refs

        raw_for_last_user: Any = None
        for raw_idx in range(len(chat_history) - 1, -1, -1):
            raw_msg = chat_history[raw_idx]
            if isinstance(raw_msg, UserMessage) or (
                isinstance(raw_msg, dict) and raw_msg.get("role") == "user"
            ):
                raw_for_last_user = raw_msg
                # Tool results also use the user role in text-only templates.
                # Locate the real user message, preserving later tool feedback.
                last_user_idx = len(
                    self._prepare_chat_messages_text_only(chat_history[:raw_idx + 1])
                ) - 1
                break

        if raw_for_last_user is None:
            last_text = messages[last_user_idx].get("content", "")
            return messages, last_text, image_refs, audio_refs

        content = raw_for_last_user.content if hasattr(raw_for_last_user, "content") else raw_for_last_user.get("content", "")
        if isinstance(content, list):
            last_text, image_refs, audio_refs = self._convert_multimodal_content_to_text_and_refs(
                content, tmp_dir
            )
        else:
            last_text = str(content)

        messages[last_user_idx] = {"role": "user", "content": last_text}
        return messages, last_text, image_refs, audio_refs

    def _build_prompt(
        self,
        messages: List[Dict[str, str]],
        apply_chat_template: Any,
        num_images: int,
        num_audios: int,
    ) -> str:
        self._ensure_loaded()
        processor = self._processor
        config = self._config
        if config is None:
            config = getattr(self._model, "config", None)
        if config is None:
            config = {}
        try:
            prompt = apply_chat_template(
                processor,
                config,
                messages,
                add_generation_prompt=True,
                num_images=num_images,
                num_audios=num_audios,
            )
        except TypeError:
            # Older mlx-vlm: string prompt style
            try:
                user_text = messages[-1].get("content", "") if messages else ""
                prompt = apply_chat_template(
                    processor,
                    config,
                    user_text,
                    add_generation_prompt=True,
                    num_images=num_images,
                    num_audios=num_audios,
                )
            except TypeError:
                try:
                    user_text = messages[-1].get("content", "") if messages else ""
                    prompt = apply_chat_template(
                        processor,
                        config,
                        user_text,
                        num_images=num_images,
                    )
                except Exception as exc:
                    raise InvalidRequestError(message=f"apply_chat_template failed: {exc}") from exc
            except Exception as exc:
                raise InvalidRequestError(message=f"apply_chat_template failed: {exc}") from exc
        except Exception as exc:
            raise InvalidRequestError(message=f"apply_chat_template failed: {exc}") from exc
        if not isinstance(prompt, str):
            raise InvalidRequestError(message="apply_chat_template did not return a string")
        return prompt

    @staticmethod
    def _generation_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        return str(result)

    def _estimate_usage(self, prompt: str, completion: str) -> LLMUsage:
        self._ensure_loaded()
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            return LLMUsage(
                prompt_tokens=max(0, len(prompt) // 4),
                completion_tokens=max(0, len(completion) // 4),
                total_tokens=max(0, (len(prompt) + len(completion)) // 4),
            )
        try:
            pt = len(tokenizer.encode(prompt))
            ct = len(tokenizer.encode(completion))
        except Exception:
            pt, ct = 0, max(0, len(completion) // 4)
        return LLMUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

    @staticmethod
    def _usage_from_generation_result(result: Any, fallback: LLMUsage) -> LLMUsage:
        pt = getattr(result, "prompt_tokens", None)
        ct = getattr(result, "generation_tokens", None)
        tt = getattr(result, "total_tokens", None)
        if isinstance(pt, int) and isinstance(ct, int) and isinstance(tt, int):
            return LLMUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
        return fallback

    def _mlx_generate_call(
        self,
        generate_fn: Any,
        prompt: str,
        image_refs: List[str],
        audio_refs: List[str],
        verbose: bool,
        extra: Dict[str, Any],
    ) -> Any:
        self._ensure_loaded()
        img = image_refs if image_refs else None
        aud = audio_refs if audio_refs else None
        return generate_fn(
            self._model,
            self._processor,
            prompt,
            img,
            audio=aud,
            verbose=verbose,
            **extra,
        )

    def _handle_error(self, error: Exception) -> Exception:
        if isinstance(error, LLMError):
            return error
        err_s = str(error).lower()
        if "cannot find" in err_s or "not found" in err_s or "no such file" in err_s:
            return ModelNotFoundError(model_id=self.model_id, provider="MLX-VLM")
        if "context" in err_s and ("length" in err_s or "long" in err_s or "exceed" in err_s):
            return ContextLengthExceededError(model_id=self.model_id, tokens=None, max_tokens=None)
        if "memory" in err_s or "oom" in err_s or "allocate" in err_s:
            return ServerError(provider="MLX-VLM", status_code=None)
        if isinstance(error, (builtins.ConnectionError, TimeoutError, OSError)):
            return ConnectionError(provider="MLX-VLM", message=str(error))
        return LLMError(message=f"MLX-VLM error: {error}")


class AsyncMLXVLMProvider(AsyncBaseProvider, _MLXVLMProviderMixin):
    """Async MLX-VLM provider."""

    def __init__(
        self,
        model_id: str,
        *,
        lazy_load: bool = True,
        require_apple_silicon: bool = True,
        **kwargs: Any,
    ) -> None:
        if require_apple_silicon:
            _require_apple_silicon()
        if not mlx_vlm_available():
            raise EnvironmentError(
                message="mlx_vlm is not installed. Install with: pip install 'seqoria-agent[mlx-vlm]'",
                missing_vars=[],
            )
        self.model_id, self.default_params, self.max_retries = self._init_common(model_id, **kwargs)
        self._model = None
        self._processor = None
        self._config = None
        if not lazy_load:
            self._ensure_loaded()

    async def async_generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del tool_choice
        if tools:
            raise InvalidRequestError(
                message="MLX-VLM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_vlm_generate_kwargs(params)
        _, generate_fn, _, apply_chat_template_fn, _ = _import_mlx_vlm()

        def _run() -> LLMResponse:
            with tempfile.TemporaryDirectory(prefix="seqoria_mlx_vlm_") as tmp_dir:
                messages, _last_text, image_refs, audio_refs = self._last_user_message_media(
                    chat_history, tmp_dir
                )
                num_images = len(image_refs)
                num_audios = len(audio_refs)
                prompt = self._build_prompt(messages, apply_chat_template_fn, num_images, num_audios)
                verbose = bool(gen_kw.get("verbose", False))
                extra = {k: v for k, v in gen_kw.items() if k != "verbose"}
                try:
                    raw = self._mlx_generate_call(
                        generate_fn, prompt, image_refs, audio_refs, verbose, extra
                    )
                except Exception as exc:
                    raise self._handle_error(exc) from exc
                text = self._generation_text(raw)
                fallback = self._estimate_usage(prompt, text)
                usage = self._usage_from_generation_result(raw, fallback)
                return LLMResponse(
                    model_id=effective_model,
                    type="llm_response",
                    text=text,
                    thinking=None,
                    usage=usage,
                    stop_reason=None,
                )

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise self._handle_error(exc) from exc

    async def async_stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream via mlx_vlm ``stream_generate`` (blocking work runs in a worker thread)."""
        del tool_choice
        if tools:
            raise InvalidRequestError(
                message="MLX-VLM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_vlm_generate_kwargs(params)
        _, _, stream_fn, apply_chat_template_fn, _ = _import_mlx_vlm()

        def _consume_stream() -> List[LLMResponse]:
            out: List[LLMResponse] = []
            with tempfile.TemporaryDirectory(prefix="seqoria_mlx_vlm_") as tmp_dir:
                messages, _lt, image_refs, audio_refs = self._last_user_message_media(chat_history, tmp_dir)
                num_images = len(image_refs)
                num_audios = len(audio_refs)
                prompt = self._build_prompt(messages, apply_chat_template_fn, num_images, num_audios)
                self._ensure_loaded()
                verbose = bool(gen_kw.get("verbose", False))
                extra = {k: v for k, v in gen_kw.items() if k != "verbose"}
                img = image_refs if image_refs else None
                aud = audio_refs if audio_refs else None
                full = ""
                last: Any = None
                try:
                    for response in stream_fn(
                        self._model,
                        self._processor,
                        prompt,
                        image=img,
                        audio=aud,
                        verbose=verbose,
                        **extra,
                    ):
                        last = response
                        piece = getattr(response, "text", "") or ""
                        if piece:
                            full += piece
                            out.append(
                                LLMResponse(
                                    model_id=effective_model,
                                    type="llm_delta",
                                    text=piece,
                                    thinking=None,
                                    usage=None,
                                    stop_reason=None,
                                )
                            )
                except Exception as exc:
                    raise self._handle_error(exc) from exc
                fallback = self._estimate_usage(prompt, full)
                usage = self._usage_from_generation_result(last, fallback) if last is not None else fallback
                out.append(
                    LLMResponse(
                        model_id=effective_model,
                        type="llm_response",
                        text=full,
                        thinking=None,
                        usage=usage,
                        stop_reason=None,
                    )
                )
            return out

        try:
            chunks = await asyncio.to_thread(_consume_stream)
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise self._handle_error(exc) from exc
        for item in chunks:
            yield item


class MLXVLMProvider(BaseProvider, _MLXVLMProviderMixin):
    """Synchronous MLX-VLM provider."""

    def __init__(
        self,
        model_id: str,
        *,
        lazy_load: bool = True,
        require_apple_silicon: bool = True,
        **kwargs: Any,
    ) -> None:
        if require_apple_silicon:
            _require_apple_silicon()
        if not mlx_vlm_available():
            raise EnvironmentError(
                message="mlx_vlm is not installed. Install with: pip install 'seqoria-agent[mlx-vlm]'",
                missing_vars=[],
            )
        self.model_id, self.default_params, self.max_retries = self._init_common(model_id, **kwargs)
        self._model = None
        self._processor = None
        self._config = None
        if not lazy_load:
            self._ensure_loaded()

    def generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del tool_choice
        if tools:
            raise InvalidRequestError(
                message="MLX-VLM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_vlm_generate_kwargs(params)
        _, generate_fn, _, apply_chat_template_fn, _ = _import_mlx_vlm()

        with tempfile.TemporaryDirectory(prefix="seqoria_mlx_vlm_") as tmp_dir:
            messages, _last_text, image_refs, audio_refs = self._last_user_message_media(chat_history, tmp_dir)
            num_images = len(image_refs)
            num_audios = len(audio_refs)
            prompt = self._build_prompt(messages, apply_chat_template_fn, num_images, num_audios)
            verbose = bool(gen_kw.get("verbose", False))
            extra = {k: v for k, v in gen_kw.items() if k != "verbose"}
            try:
                raw = self._mlx_generate_call(
                    generate_fn, prompt, image_refs, audio_refs, verbose, extra
                )
            except Exception as exc:
                raise self._handle_error(exc) from exc
            text = self._generation_text(raw)
            fallback = self._estimate_usage(prompt, text)
            usage = self._usage_from_generation_result(raw, fallback)
            return LLMResponse(
                model_id=effective_model,
                type="llm_response",
                text=text,
                thinking=None,
                usage=usage,
                stop_reason=None,
            )

    def stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[LLMResponse, None, None]:
        del tool_choice
        if tools:
            raise InvalidRequestError(
                message="MLX-VLM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_vlm_generate_kwargs(params)
        _, _, stream_fn, apply_chat_template_fn, _ = _import_mlx_vlm()

        with tempfile.TemporaryDirectory(prefix="seqoria_mlx_vlm_") as tmp_dir:
            messages, _lt, image_refs, audio_refs = self._last_user_message_media(chat_history, tmp_dir)
            num_images = len(image_refs)
            num_audios = len(audio_refs)
            prompt = self._build_prompt(messages, apply_chat_template_fn, num_images, num_audios)
            verbose = bool(gen_kw.get("verbose", False))
            extra = {k: v for k, v in gen_kw.items() if k != "verbose"}
            img = image_refs if image_refs else None
            aud = audio_refs if audio_refs else None
            full = ""
            last: Any = None
            try:
                for response in stream_fn(
                    self._model,
                    self._processor,
                    prompt,
                    image=img,
                    audio=aud,
                    verbose=verbose,
                    **extra,
                ):
                    last = response
                    piece = getattr(response, "text", "") or ""
                    if piece:
                        full += piece
                        yield LLMResponse(
                            model_id=effective_model,
                            type="llm_delta",
                            text=piece,
                            thinking=None,
                            usage=None,
                            stop_reason=None,
                        )
            except Exception as exc:
                raise self._handle_error(exc) from exc
            fallback = self._estimate_usage(prompt, full)
            usage = self._usage_from_generation_result(last, fallback) if last is not None else fallback
            yield LLMResponse(
                model_id=effective_model,
                type="llm_response",
                text=full,
                thinking=None,
                usage=usage,
                stop_reason=None,
            )
