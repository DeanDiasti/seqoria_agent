"""
Local LLM provider using MLX-LM on Apple Silicon (mlx_lm).

Loads models via Hugging Face repo id or local path; uses ``tokenizer.apply_chat_template``
and ``mlx_lm.generate`` / ``stream_generate``. Blocking work runs in ``asyncio.to_thread``
for async providers.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import platform
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

from seqoria_agent.client.ai_client.utils import json_dumps, json_loads
from seqoria_agent.models.chunk import (
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolCallResult,
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

# Generation kwargs forwarded to mlx_lm.generate / stream_generate (after normalization).
_MLX_GEN_KEYS = frozenset(
    {
        "max_tokens",
        "temp",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "verbose",
        "repetition_penalty",
        "repetition_context_size",
        "logits_processors",
        "sampler",
        "eos_token_id",
    }
)


def mlx_lm_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise EnvironmentError(
            message="MLX-LM requires macOS on Apple Silicon (arm64).",
            missing_vars=[],
        )


def _import_mlx_lm() -> Tuple[Any, Any, Any]:
    try:
        from mlx_lm import generate, load, stream_generate
    except ImportError as exc:
        raise EnvironmentError(
            message="mlx_lm is not installed. Install with: pip install 'seqoria-agent[mlx-lm]'",
            missing_vars=[],
        ) from exc
    return load, generate, stream_generate


class _MLXLMProviderMixin:
    """Shared logic for MLX-LM sync and async providers."""

    model_id: str
    max_retries: int
    default_params: Dict[str, Any]
    _load_fn: Any
    _generate_fn: Any
    _stream_generate_fn: Any
    _model: Any
    _tokenizer: Any

    def _init_common(self, model_id: str, **kwargs: Any) -> Tuple[str, Dict[str, Any], int]:
        if not model_id or not str(model_id).strip():
            raise InvalidRequestError(message="model_id is required for MLX-LM provider")
        max_retries = int(kwargs.get("max_retries", 3))
        default_params: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _MLX_GEN_KEYS and key != "max_retries":
                default_params[key] = value
        return str(model_id), default_params, max_retries

    def _prepare_params(self, **kwargs: Any) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update(kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def _mlx_generate_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map merged params to mlx_lm.generate keyword arguments."""
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if k in _MLX_GEN_KEYS or k == "temperature":
                out[k] = v
        if "temperature" in out and "temp" not in out:
            out["temp"] = out.pop("temperature")
        elif "temperature" in out:
            out.pop("temperature", None)
        return out

    def _ensure_loaded(self) -> None:
        if getattr(self, "_model", None) is not None and getattr(self, "_tokenizer", None) is not None:
            return
        load_fn, _, _ = _import_mlx_lm()
        adapter_path = getattr(self, "_adapter_path", None)
        kwargs = getattr(self, "_load_kwargs", {}) or {}
        try:
            if adapter_path:
                self._model, self._tokenizer = load_fn(self.model_id, adapter_path=adapter_path, **kwargs)
            else:
                self._model, self._tokenizer = load_fn(self.model_id, **kwargs)
        except Exception as exc:
            raise self._handle_error(exc) from exc

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

    def _convert_image_content_text_only(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """Multimodal blocks to plain text (mlx-lm does not consume images here)."""
        if isinstance(content, str):
            return content
        text_parts: List[str] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype in ("image", "image_url"):
                text_parts.append("[image]")
        return " ".join(text_parts).strip()

    def _prepare_chat_messages(
        self,
        chat_history: List[
            Union[Dict[str, Any], ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage]
        ],
    ) -> List[Dict[str, str]]:
        """Normalize history to role/content strings for ``apply_chat_template``."""
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

            if isinstance(raw_msg, ToolCallResult):
                prepared.append(
                    {
                        "role": "user",
                        "content": f"[tool_result] {raw_msg.name}: {raw_msg.result}",
                    }
                )
                continue

            if isinstance(raw_msg, (AssistantMessage, UserMessage, SystemMessage)):
                text_content = self._convert_image_content_text_only(raw_msg.content)
                prepared.append({"role": str(raw_msg.role), "content": text_content})
                continue

            if not isinstance(raw_msg, dict):
                continue

            role = raw_msg.get("role")
            content = raw_msg.get("content", "")

            if role == "tool_call":
                prepared.append(
                    {
                        "role": "assistant",
                        "content": f"[tool_request] {raw_msg.get('name')} {json_dumps(raw_msg.get('arguments', {}))}",
                    }
                )
                continue

            if role in {"tool_result", "tool"}:
                prepared.append(
                    {
                        "role": "user",
                        "content": f"[tool_result] {raw_msg.get('name', '')}: {raw_msg.get('result', content)}",
                    }
                )
                continue

            if role in {"assistant", "system", "user"}:
                text_content = self._convert_image_content_text_only(content)
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

        return prepared

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:
            raise InvalidRequestError(message=f"apply_chat_template failed: {exc}") from exc
        if not isinstance(prompt, str):
            raise InvalidRequestError(message="tokenizer.apply_chat_template did not return a string")
        return prompt

    def _estimate_usage(self, prompt: str, completion: str) -> LLMUsage:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        try:
            pt = len(tokenizer.encode(prompt))
            ct = len(tokenizer.encode(completion))
        except Exception:
            pt, ct = 0, max(0, len(completion) // 4)
        return LLMUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

    def _handle_error(self, error: Exception) -> Exception:
        if isinstance(error, LLMError):
            return error
        err_s = str(error).lower()
        if "cannot find" in err_s or "not found" in err_s or "no such file" in err_s:
            return ModelNotFoundError(model_id=self.model_id, provider="MLX-LM")
        if "context" in err_s and ("length" in err_s or "long" in err_s or "exceed" in err_s):
            return ContextLengthExceededError(model_id=self.model_id, tokens=None, max_tokens=None)
        if "memory" in err_s or "oom" in err_s or "allocate" in err_s:
            return ServerError(provider="MLX-LM", status_code=None)
        if isinstance(error, (builtins.ConnectionError, TimeoutError, OSError)):
            return ConnectionError(provider="MLX-LM", message=str(error))
        return LLMError(message=f"MLX-LM error: {error}")


class AsyncMLXLMProvider(AsyncBaseProvider, _MLXLMProviderMixin):
    """Async MLX-LM provider: runs generation in ``asyncio.to_thread``."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Optional[str] = None,
        lazy_load: bool = True,
        require_apple_silicon: bool = True,
        **kwargs: Any,
    ) -> None:
        if require_apple_silicon:
            _require_apple_silicon()
        if not mlx_lm_available():
            raise EnvironmentError(
                message="mlx_lm is not installed. Install with: pip install 'seqoria-agent[mlx-lm]'",
                missing_vars=[],
            )
        self._adapter_path = adapter_path
        self._load_kwargs = {k: v for k, v in kwargs.items() if k in ("draft_model", "tokenizer_config", "model_config")}
        self.model_id, self.default_params, self.max_retries = self._init_common(model_id, **kwargs)
        self._model = None
        self._tokenizer = None
        load_fn, gen_fn, stream_fn = _import_mlx_lm()
        self._load_fn = load_fn
        self._generate_fn = gen_fn
        self._stream_generate_fn = stream_fn
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
                message="MLX-LM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_generate_kwargs(params)
        messages = self._prepare_chat_messages(chat_history)

        def _run() -> LLMResponse:
            self._ensure_loaded()
            prompt = self._build_prompt(messages)
            try:
                raw = self._generate_fn(self._model, self._tokenizer, prompt=prompt, **gen_kw)
            except Exception as exc:
                raise self._handle_error(exc) from exc
            text = self._generation_text(raw)
            usage = self._estimate_usage(prompt, text)
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
        del tool_choice
        if tools:
            raise InvalidRequestError(
                message="MLX-LM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_generate_kwargs(params)
        messages = self._prepare_chat_messages(chat_history)

        def _consume_stream() -> List[LLMResponse]:
            out: List[LLMResponse] = []
            self._ensure_loaded()
            prompt = self._build_prompt(messages)
            full = ""
            try:
                for chunk in self._stream_generate_fn(self._model, self._tokenizer, prompt, **gen_kw):
                    piece = self._generation_text(chunk)
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
            usage = self._estimate_usage(prompt, full)
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


class MLXLMProvider(BaseProvider, _MLXLMProviderMixin):
    """Synchronous MLX-LM provider."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Optional[str] = None,
        lazy_load: bool = True,
        require_apple_silicon: bool = True,
        **kwargs: Any,
    ) -> None:
        if require_apple_silicon:
            _require_apple_silicon()
        if not mlx_lm_available():
            raise EnvironmentError(
                message="mlx_lm is not installed. Install with: pip install 'seqoria-agent[mlx-lm]'",
                missing_vars=[],
            )
        self._adapter_path = adapter_path
        self._load_kwargs = {k: v for k, v in kwargs.items() if k in ("draft_model", "tokenizer_config", "model_config")}
        self.model_id, self.default_params, self.max_retries = self._init_common(model_id, **kwargs)
        self._model = None
        self._tokenizer = None
        load_fn, gen_fn, stream_fn = _import_mlx_lm()
        self._load_fn = load_fn
        self._generate_fn = gen_fn
        self._stream_generate_fn = stream_fn
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
                message="MLX-LM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_generate_kwargs(params)
        messages = self._prepare_chat_messages(chat_history)
        self._ensure_loaded()
        prompt = self._build_prompt(messages)
        try:
            raw = self._generate_fn(self._model, self._tokenizer, prompt=prompt, **gen_kw)
        except Exception as exc:
            raise self._handle_error(exc) from exc
        text = self._generation_text(raw)
        usage = self._estimate_usage(prompt, text)
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
                message="MLX-LM provider does not support tool calling; use an HTTP provider or remove tools.",
            )
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        gen_kw = self._mlx_generate_kwargs(params)
        messages = self._prepare_chat_messages(chat_history)
        self._ensure_loaded()
        prompt = self._build_prompt(messages)
        full = ""
        try:
            for chunk in self._stream_generate_fn(self._model, self._tokenizer, prompt, **gen_kw):
                piece = self._generation_text(chunk)
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
        usage = self._estimate_usage(prompt, full)
        yield LLMResponse(
            model_id=effective_model,
            type="llm_response",
            text=full,
            thinking=None,
            usage=usage,
            stop_reason=None,
        )
