"""OpenAI-compatible providers using direct HTTPX Chat Completions calls."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import httpx

from seqoria_agent._redaction import safe_exception_cause
from seqoria_agent.client.ai_client.utils import json_dumps, json_loads
from seqoria_agent.models.chunk import (
    AssistantMessage,
    SystemMessage,
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
    UserMessage,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider, ChatHistory
from seqoria_agent.providers.errors import (
    ConnectionError,
    ContentFilterError,
    ContextLengthExceededError,
    EnvironmentError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    QuotaExceededError,
    ServerError,
)
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList

from ._sse import aiter_sse_data, iter_sse_data
from ._validation import parse_tool_arguments, validate_max_retries

_EMPTY_JSON: str = "{}"


class _OpenAICompatibleProviderMixin:
    """Shared logic for sync/async OpenAI-compatible chat completions providers."""

    default_params: Dict[str, Any]
    model_id: Optional[str]

    def _init_common(
        self, model_id: Optional[str], api_key: Optional[str], base_url: Optional[str], **kwargs
    ) -> tuple[str, Any, str, Dict[str, Any], int, Dict[str, str]]:
        """Initialize common provider fields and validated defaults."""
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                message="OpenAI API key not provided", missing_vars=["OPENAI_API_KEY"]
            )

        if not model_id:
            raise InvalidRequestError(message="model_id is required for OpenAI-compatible provider")

        effective_base_url = (
            base_url or kwargs.get("base_url") or "https://api.openai.com/v1"
        ).rstrip("/")
        timeout = kwargs.get("timeout", 30.0)
        max_retries = validate_max_retries(kwargs.get("max_retries", 3))

        # Chat Completions request-body parameters (excluding model/messages/stream).
        valid_query_keys = {
            "audio",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "metadata",
            "modalities",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "prompt_cache_key",
            "prompt_cache_retention",
            "reasoning_effort",
            "response_format",
            "safety_identifier",
            "seed",
            "service_tier",
            "stop",
            "store",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "verbosity",
            "web_search_options",
        }

        default_params: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return effective_base_url, timeout, model_id, default_params, max_retries, headers

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """Merge default params with call-level overrides, dropping None values."""
        params = self.default_params.copy()
        params.update(kwargs)
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def _make_strict_json_schema(schema: Any) -> Any:  # noqa: ANN401
        """Return a strict-compliant JSON schema copy for function parameters."""
        if isinstance(schema, list):
            return [_OpenAICompatibleProviderMixin._make_strict_json_schema(x) for x in schema]
        if not isinstance(schema, dict):
            return schema

        out: Dict[str, Any] = {
            k: _OpenAICompatibleProviderMixin._make_strict_json_schema(v) for k, v in schema.items()
        }
        if out.get("type") == "object":
            props = out.get("properties")
            if isinstance(props, dict) and props:
                out["additionalProperties"] = False
        return out

    def _normalize_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Normalize shorthand tool-choice payloads to Chat Completions format."""
        if not isinstance(tool_choice, dict):
            return tool_choice

        if (
            tool_choice.get("type") == "function"
            and "name" in tool_choice
            and "function" not in tool_choice
        ):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        if (
            tool_choice.get("type") == "custom"
            and "name" in tool_choice
            and "custom" not in tool_choice
        ):
            return {"type": "custom", "custom": {"name": tool_choice["name"]}}
        return tool_choice

    def _format_tools_chat(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools into Chat Completions tool schema."""
        formatted: List[Dict[str, Any]] = []

        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type != "function":
                formatted.append(tool)
                continue

            func_def = tool.get("function", tool)
            parameters = func_def.get("parameters")
            strict = func_def.get("strict")

            if isinstance(parameters, dict) and parameters.get("type") == "object":
                props = parameters.get("properties")
                if isinstance(props, dict) and props:
                    parameters = self._make_strict_json_schema(parameters)
                elif strict is True:
                    # Mapping-style object schemas are incompatible with strict mode.
                    strict = False

            fn_payload: Dict[str, Any] = {
                "name": func_def.get("name"),
                "description": func_def.get("description", ""),
            }
            if parameters is not None:
                fn_payload["parameters"] = parameters
            if strict is not None:
                fn_payload["strict"] = strict

            formatted.append({"type": "function", "function": fn_payload})

        return formatted

    def _convert_content_blocks_chat(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert common multimodal blocks to Chat Completions content blocks."""
        converted: List[Dict[str, Any]] = []

        for block in content:
            block_type = block.get("type")

            if block_type in {"text", "input_text", "output_text"}:
                converted.append({"type": "text", "text": block.get("text", "")})
                continue

            if block_type == "image":
                if "url" in block:
                    converted.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": block["url"],
                                "detail": block.get("detail", "auto"),
                            },
                        }
                    )
                elif "base64" in block:
                    media_type = block.get("media_type", "image/jpeg")
                    data_url = f"data:{media_type};base64,{block['base64']}"
                    converted.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": block.get("detail", "auto"),
                            },
                        }
                    )
                continue

            if block_type == "input_image":
                image_url = block.get("image_url", "")
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": block.get("detail", "auto")},
                    }
                )
                continue

            if block_type == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    converted.append({"type": "image_url", "image_url": image_url})
                else:
                    converted.append({"type": "image_url", "image_url": {"url": image_url}})
                continue

            # Keep valid chat-completions blocks as-is.
            if block_type in {"input_audio", "file", "refusal"}:
                converted.append(block)
                continue

            # Fallback: try preserving text payload.
            if "text" in block:
                converted.append({"type": "text", "text": block.get("text", "")})
            else:
                converted.append(block)

        return converted

    def _build_function_tool_call(self, call_id: Any, name: str, arguments: Any) -> Dict[str, Any]:  # noqa: ANN401
        """Build one assistant tool-call entry in Chat Completions schema."""
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json_dumps(arguments or {})

        return {
            "id": str(call_id or ""),
            "type": "function",
            "function": {
                "name": name or "",
                "arguments": arguments_json if arguments_json else _EMPTY_JSON,
            },
        }

    def _normalize_assistant_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:  # noqa: ANN401
        """Normalize arbitrary tool-call payloads into Chat Completions schema."""
        if not isinstance(tool_calls, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            function = tc.get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments", _EMPTY_JSON)
                if isinstance(arguments, (dict, list)):
                    arguments = json_dumps(arguments)
                normalized.append(
                    {
                        "id": str(tc.get("id", "")),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": function.get("name", tc.get("name", "")),
                            "arguments": arguments if arguments else _EMPTY_JSON,
                        },
                    }
                )
                continue

            # Shorthand call structure.
            normalized.append(
                self._build_function_tool_call(
                    call_id=tc.get("id", tc.get("call_id")),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                )
            )

        return normalized

    def _prepare_messages_chat(
        self,
        chat_history: ChatHistory,
    ) -> List[Dict[str, Any]]:
        """Prepare chat history in Chat Completions message format."""
        prepared: List[Dict[str, Any]] = []

        for raw_msg in chat_history:
            if isinstance(raw_msg, ToolCallRequest):
                prepared.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            self._build_function_tool_call(
                                call_id=raw_msg.id,
                                name=raw_msg.name,
                                arguments=raw_msg.arguments,
                            )
                        ],
                    }
                )
                continue

            if isinstance(raw_msg, (ToolCallResult, ToolCallError)):
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(raw_msg.tool_call_id),
                        "content": str(
                            raw_msg.error if isinstance(raw_msg, ToolCallError) else raw_msg.result
                        ),
                    }
                )
                continue

            if isinstance(raw_msg, (SystemMessage, AssistantMessage, UserMessage)):
                content = raw_msg.content
                content_payload: Union[str, List[Dict[str, Any]], None]
                if isinstance(content, list):
                    content_payload = self._convert_content_blocks_chat(content)
                else:
                    content_payload = content
                prepared.append({"role": raw_msg.role, "content": content_payload})
                continue

            role = raw_msg.get("role")

            if role in {"system", "developer", "user", "assistant"}:
                content = raw_msg.get("content")
                if isinstance(content, list):
                    content = self._convert_content_blocks_chat(content)

                msg: Dict[str, Any] = {"role": role, "content": content}

                if role == "assistant" and raw_msg.get("tool_calls") is not None:
                    msg["tool_calls"] = self._normalize_assistant_tool_calls(
                        raw_msg.get("tool_calls")
                    )
                    if "content" not in raw_msg:
                        msg["content"] = None
                if role == "assistant" and raw_msg.get("refusal") is not None:
                    msg["refusal"] = raw_msg.get("refusal")

                prepared.append(msg)
                continue

            if role == "tool_call":
                prepared.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            self._build_function_tool_call(
                                call_id=raw_msg.get("id", raw_msg.get("call_id")),
                                name=raw_msg.get("name", ""),
                                arguments=raw_msg.get("arguments", {}),
                            )
                        ],
                    }
                )
                continue

            if role in {"tool_result", "tool_error", "tool"}:
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(raw_msg.get("tool_call_id", raw_msg.get("id", ""))),
                        "content": str(
                            raw_msg.get("error", "")
                            if role == "tool_error"
                            else raw_msg.get("result", raw_msg.get("content", ""))
                        ),
                    }
                )
                continue

            if role == "function":
                prepared.append(
                    {
                        "role": "function",
                        "name": raw_msg.get("name"),
                        "content": str(raw_msg.get("content", "")),
                    }
                )

        return prepared

    def _extract_text_from_message(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a Chat Completions assistant message."""
        content = message.get("content")
        if isinstance(content, str):
            return content

        parts: List[str] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"text", "output_text"} and part.get("text"):
                    parts.append(part["text"])
                elif part_type == "refusal" and part.get("refusal"):
                    parts.append(part["refusal"])

        if parts:
            return "".join(parts)

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            return refusal
        return None

    @staticmethod
    def _extract_reasoning_text(obj: Dict[str, Any]) -> Optional[str]:
        """Extract reasoning/thinking text from a Chat Completions message or delta.

        OpenRouter (and some other providers) surface reasoning tokens via:
        - ``reasoning`` (str) — plain-text reasoning content
        - ``reasoning_content`` (str) — alias used by some providers (e.g. DeepSeek)
        - ``reasoning_details`` (list[dict]) — structured reasoning detail objects
          with types ``reasoning.text`` and ``reasoning.summary``

        Some providers send both ``reasoning`` and ``reasoning_details`` in the
        same chunk with identical content.  To avoid duplication we prefer the
        structured ``reasoning_details`` when present and fall back to the
        plain-text field only when the structured array is absent or empty.
        """
        details_parts: List[str] = []
        reasoning_details = obj.get("reasoning_details")
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                if not isinstance(detail, dict):
                    continue
                detail_type = detail.get("type", "")
                if detail_type == "reasoning.text":
                    text = detail.get("text")
                    if text:
                        details_parts.append(text)
                elif detail_type == "reasoning.summary":
                    summary = detail.get("summary")
                    if summary:
                        details_parts.append(summary)

        if details_parts:
            return "".join(details_parts)

        reasoning_str = obj.get("reasoning") or obj.get("reasoning_content")
        if isinstance(reasoning_str, str) and reasoning_str:
            return reasoning_str

        return None

    def _extract_usage(self, payload: Dict[str, Any]) -> LLMUsage:
        """Extract token usage from Chat Completions payload."""
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _parse_tool_call_sync(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        """Parse Chat Completions tool calls into standardized ToolCall objects."""
        parsed: List[ToolCall] = []

        for tc in tool_calls:
            function = tc.get("function", {}) if isinstance(tc, dict) else {}
            args_raw = function.get("arguments", _EMPTY_JSON)

            arguments = parse_tool_arguments(args_raw)

            parsed.append(
                ToolCall(
                    id=str(tc.get("id", "")),
                    type="function_call",
                    function_name=function.get("name", tc.get("name", "")),
                    arguments=arguments or {},
                )
            )

        return parsed

    def _extract_error_details_from_status_error(
        self, error: httpx.HTTPStatusError
    ) -> tuple[int, str, Optional[str], Optional[str]]:
        """Extract status, message, code, and param from an HTTP status error."""
        response = error.response
        status_code = response.status_code
        message = str(error)
        code = None
        param = None

        def _safe_response_text(max_chars: int = 1200) -> Optional[str]:
            try:
                text = response.text
            except Exception:
                try:
                    response.read()
                    text = response.text
                except Exception:
                    return None
            cleaned = text.strip()
            if not cleaned:
                return None
            if len(cleaned) > max_chars:
                return f"{cleaned[:max_chars]}..."
            return cleaned

        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error", {})
                if isinstance(err, dict):
                    message = err.get("message", message)
                    code = err.get("code")
                    param = err.get("param")
                elif isinstance(err, str) and err.strip():
                    message = err
        except Exception:
            pass

        if message == str(error) or message.lower().startswith(("client error", "server error")):
            raw_body = _safe_response_text()
            if raw_body:
                message = raw_body

        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if request_id and f"request_id={request_id}" not in message:
            message = f"{message} (request_id={request_id})"
        return status_code, message, code, param

    def _handle_error(self, error: Exception) -> Exception:
        """Map HTTPX/OpenAI-compatible errors into provider error taxonomy."""
        if isinstance(error, LLMError):
            return error

        if isinstance(error, httpx.HTTPStatusError):
            status_code, message, code, param = self._extract_error_details_from_status_error(error)
            message_l = message.lower()
            code_l = str(code or "").lower()

            if status_code == 401:
                return EnvironmentError(
                    message=f"OpenAI authentication failed: {message}",
                    missing_vars=["OPENAI_API_KEY"],
                )
            if status_code == 429:
                return QuotaExceededError(provider="OpenAI", limit_type="rate")
            if status_code == 404:
                if code_l == "model_not_found" or "model" in message_l:
                    return ModelNotFoundError(model_id=self.model_id, provider="OpenAI")
                return InvalidRequestError(message=message, param=param)
            if status_code == 400:
                if "context_length_exceeded" in code_l or "maximum context length" in message_l:
                    nums = [int(x) for x in re.findall(r"\d+", message)]
                    tokens = nums[0] if nums else None
                    max_tokens = nums[1] if len(nums) > 1 else None
                    return ContextLengthExceededError(
                        model_id=self.model_id,
                        tokens=tokens,
                        max_tokens=max_tokens,
                    )
                if "content_policy_violation" in code_l:
                    return ContentFilterError(provider="OpenAI", filter_reason=message)
                return InvalidRequestError(message=message, param=param)
            if status_code == 403:
                return InvalidRequestError(
                    message=f"OpenAI permission denied: {message}", param=param
                )
            if status_code == 422:
                return ContentFilterError(provider="OpenAI", filter_reason=message)
            if 500 <= status_code < 600:
                return ServerError(provider="OpenAI", status_code=status_code)
            return LLMError(message=f"OpenAI API Error {status_code}: {message}")

        if isinstance(
            error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)
        ):
            return ConnectionError(provider="OpenAI", message="Request timed out")
        if isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).lower():
            return ConnectionError(provider="OpenAI", message="Request timed out")
        if isinstance(error, (httpx.NetworkError, httpx.TransportError, httpx.ConnectError)):
            return ConnectionError(provider="OpenAI", message=f"Connection error: {error}")
        if isinstance(error, httpx.HTTPError):
            return ConnectionError(provider="OpenAI", message=f"HTTP error: {error}")
        return LLMError(message=f"An unexpected OpenAI error occurred: {error}")

    def _validate_model_sync(self, client: httpx.Client, model_id: str) -> None:
        """Validate a model ID against `/models` endpoint."""
        try:
            response = client.get("/models")
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            available_models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="OpenAI",
                    available_models=available_models,
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            mapped_error = self._handle_error(e)
            raise mapped_error from safe_exception_cause(e, raised=mapped_error)

    async def _validate_model_async(self, client: httpx.AsyncClient, model_id: str) -> None:
        """Asynchronously validate a model ID against `/models` endpoint."""
        try:
            response = await client.get("/models")
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            available_models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="OpenAI",
                    available_models=available_models,
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            mapped_error = self._handle_error(e)
            raise mapped_error from safe_exception_cause(e, raised=mapped_error)


class AsyncOpenAICompatibleProvider(AsyncBaseProvider, _OpenAICompatibleProviderMixin):
    """Async OpenAI-compatible provider using HTTPX + Chat Completions API."""

    model_id: str

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        (
            self.base_url,
            self.timeout,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.headers,
        ) = self._init_common(model_id, api_key, base_url, **kwargs)

        try:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="OpenAI", message=str(e)) from safe_exception_cause(e)

    async def validate_model(self) -> None:
        """Validate that configured model exists for this endpoint."""
        await self._validate_model_async(self.async_client, self.model_id)

    async def _parse_tool_call(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        return self._parse_tool_call_sync(tool_calls)

    async def async_generate(
        self,
        chat_history: ChatHistory,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        params = self._prepare_params(**kwargs)
        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages_chat(chat_history),
            **params,
        }

        if tools is not None:
            payload["tools"] = self._format_tools_chat(tools)
        if tool_choice is not None:
            payload["tool_choice"] = self._normalize_tool_choice(tool_choice)

        async def _generate() -> Union[LLMResponse, ToolCallList]:
            response = await self.async_client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise LLMError("Provider returned a malformed response: expected a JSON object")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMError(message="Provider returned a malformed response: expected choices")

            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise LLMError("Provider returned a malformed response: expected a message object")
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            usage_obj = self._extract_usage(data)
            model_used = data.get("model", effective_model)

            if tool_calls:
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise LLMError("Provider response ended before tool calls completed")
                tool_calls_parsed = await self._parse_tool_call(tool_calls)
                return ToolCallList(
                    model_id=model_used, tool_calls=tool_calls_parsed, usage=usage_obj
                )

            text = self._extract_text_from_message(message)
            thinking = self._extract_reasoning_text(message)
            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text,
                thinking=thinking,
                usage=usage_obj,
                stop_reason=choice.get("finish_reason"),
            )

        return await self._execute_with_retry(operation=_generate)

    async def async_stream(
        self,
        chat_history: ChatHistory,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        params = self._prepare_params(**kwargs)
        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages_chat(chat_history),
            "stream": True,
            **params,
        }

        if tools is not None:
            payload["tools"] = self._format_tools_chat(tools)
        if tool_choice is not None:
            payload["tool_choice"] = self._normalize_tool_choice(tool_choice)

        # Ask for final usage chunk unless caller already provided stream_options.
        if "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}

        async def _open_stream():
            stream_ctx = self.async_client.stream("POST", "/chat/completions", json=payload)
            response = await stream_ctx.__aenter__()
            try:
                if response.status_code >= 400:
                    try:
                        await response.aread()
                    except Exception:
                        pass
                response.raise_for_status()
            except BaseException as e:
                await stream_ctx.__aexit__(type(e), e, e.__traceback__)
                raise
            return stream_ctx, response

        completion_parts: List[str] = []
        thinking_parts: List[str] = []
        final_model = effective_model
        stop_reason: Optional[str] = None
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        stream_completed = False

        try:
            stream_ctx, response = await self._execute_with_retry(operation=_open_stream)
            try:
                async for data_str in aiter_sse_data(response.aiter_lines()):
                    data_str = data_str.strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        stream_completed = True
                        break

                    try:
                        chunk = json_loads(data_str)
                    except (ValueError, TypeError) as exc:
                        raise LLMError("Provider returned malformed stream event JSON") from exc
                    if not isinstance(chunk, dict):
                        raise LLMError("Provider stream event must be a JSON object")
                    if chunk.get("error"):
                        raise LLMError("Provider returned a streaming error")
                    if "choices" in chunk and not isinstance(chunk["choices"], list):
                        raise LLMError(
                            "Provider returned a malformed stream event: expected choices"
                        )
                    if "choices" not in chunk and not isinstance(chunk.get("usage"), dict):
                        raise LLMError(
                            "Provider returned a malformed stream event: expected choices or usage"
                        )

                    final_model = chunk.get("model", final_model)

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage_data["prompt_tokens"] = (
                            chunk_usage.get("prompt_tokens", usage_data["prompt_tokens"])
                            or usage_data["prompt_tokens"]
                        )
                        usage_data["completion_tokens"] = (
                            chunk_usage.get("completion_tokens", usage_data["completion_tokens"])
                            or usage_data["completion_tokens"]
                        )
                        usage_data["total_tokens"] = chunk_usage.get("total_tokens") or (
                            usage_data["prompt_tokens"] + usage_data["completion_tokens"]
                        )

                    for choice in chunk.get("choices", []) or []:
                        delta = choice.get("delta", {}) or {}

                        delta_thinking = self._extract_reasoning_text(delta)
                        if delta_thinking:
                            thinking_parts.append(delta_thinking)
                            yield LLMResponse(
                                model_id=final_model,
                                type="llm_delta",
                                text=None,
                                thinking=delta_thinking,
                                usage=None,
                                stop_reason=None,
                            )

                        delta_content = delta.get("content")
                        if isinstance(delta_content, str) and delta_content:
                            completion_parts.append(delta_content)
                            yield LLMResponse(
                                model_id=final_model,
                                type="llm_delta",
                                text=delta_content,
                                thinking=None,
                                usage=None,
                                stop_reason=None,
                            )
                        elif isinstance(delta_content, list):
                            text_fragments: List[str] = []
                            for part in delta_content:
                                if isinstance(part, dict) and part.get("type") in {
                                    "text",
                                    "output_text",
                                }:
                                    text_value = part.get("text")
                                    if text_value:
                                        text_fragments.append(text_value)
                            if text_fragments:
                                merged = "".join(text_fragments)
                                completion_parts.append(merged)
                                yield LLMResponse(
                                    model_id=final_model,
                                    type="llm_delta",
                                    text=merged,
                                    thinking=None,
                                    usage=None,
                                    stop_reason=None,
                                )

                        for tc in delta.get("tool_calls", []) or []:
                            idx = int(tc.get("index", len(tool_calls_by_index)))
                            entry = tool_calls_by_index.setdefault(
                                idx,
                                {"id": "", "function_name": "", "arguments_parts": []},
                            )
                            if tc.get("id"):
                                entry["id"] = tc["id"]

                            function = tc.get("function", {}) if isinstance(tc, dict) else {}
                            if function.get("name"):
                                entry["function_name"] = function["name"]
                            arguments_delta = function.get("arguments")
                            if isinstance(arguments_delta, str) and arguments_delta:
                                entry["arguments_parts"].append(arguments_delta)

                        if choice.get("finish_reason") is not None:
                            stop_reason = choice.get("finish_reason")
                            stream_completed = True
            finally:
                await stream_ctx.__aexit__(None, None, None)

            if not stream_completed:
                raise LLMError("Provider stream ended without a completion event")
            if tool_calls_by_index and stop_reason in {"length", "content_filter"}:
                raise LLMError("Provider response ended before tool calls completed")
            usage_obj = LLMUsage(**usage_data)
            aggregated_tool_calls: List[Dict[str, Any]] = []
            for idx in sorted(tool_calls_by_index.keys()):
                entry = tool_calls_by_index[idx]
                aggregated_tool_calls.append(
                    {
                        "id": entry["id"] or str(idx),
                        "function": {
                            "name": entry["function_name"],
                            "arguments": "".join(entry["arguments_parts"]),
                        },
                    }
                )

            if aggregated_tool_calls:
                tool_calls_obj = await self._parse_tool_call(aggregated_tool_calls)
                yield ToolCallList(model_id=final_model, tool_calls=tool_calls_obj, usage=usage_obj)
            else:
                yield LLMResponse(
                    model_id=final_model,
                    type="llm_response",
                    text="".join(completion_parts) if completion_parts else None,
                    thinking="".join(thinking_parts) if thinking_parts else None,
                    usage=usage_obj,
                    stop_reason=stop_reason,
                )
        except Exception as e:
            mapped_error = self._handle_error(e)
            raise mapped_error from safe_exception_cause(e, raised=mapped_error)


class OpenAICompatibleProvider(BaseProvider, _OpenAICompatibleProviderMixin):
    """Sync OpenAI-compatible provider using HTTPX + Chat Completions API."""

    model_id: str

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        (
            self.base_url,
            self.timeout,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.headers,
        ) = self._init_common(model_id, api_key, base_url, **kwargs)

        try:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="OpenAI", message=str(e)) from safe_exception_cause(e)

        try:
            self._validate_model_sync(self.client, self.model_id)
        except BaseException:
            # Construction failed, so no caller can close the owned client.
            self.client.close()
            raise

    def validate_model(self) -> None:
        """Validate that configured model exists for this endpoint."""
        self._validate_model_sync(self.client, self.model_id)

    def generate(
        self,
        chat_history: ChatHistory,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        params = self._prepare_params(**kwargs)
        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages_chat(chat_history),
            **params,
        }

        if tools is not None:
            payload["tools"] = self._format_tools_chat(tools)
        if tool_choice is not None:
            payload["tool_choice"] = self._normalize_tool_choice(tool_choice)

        def _generate() -> Union[LLMResponse, ToolCallList]:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise LLMError("Provider returned a malformed response: expected a JSON object")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMError(message="Provider returned a malformed response: expected choices")

            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise LLMError("Provider returned a malformed response: expected a message object")
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            usage_obj = self._extract_usage(data)
            model_used = data.get("model", effective_model)

            if tool_calls:
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise LLMError("Provider response ended before tool calls completed")
                tool_calls_parsed = self._parse_tool_call_sync(tool_calls)
                return ToolCallList(
                    model_id=model_used, tool_calls=tool_calls_parsed, usage=usage_obj
                )

            text = self._extract_text_from_message(message)
            thinking = self._extract_reasoning_text(message)
            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text,
                thinking=thinking,
                usage=usage_obj,
                stop_reason=choice.get("finish_reason"),
            )

        return self._execute_with_retry(operation=_generate)

    def stream(
        self,
        chat_history: ChatHistory,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        params = self._prepare_params(**kwargs)
        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages_chat(chat_history),
            "stream": True,
            **params,
        }

        if tools is not None:
            payload["tools"] = self._format_tools_chat(tools)
        if tool_choice is not None:
            payload["tool_choice"] = self._normalize_tool_choice(tool_choice)
        if "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}

        def _open_stream():
            stream_ctx = self.client.stream("POST", "/chat/completions", json=payload)
            response = stream_ctx.__enter__()
            try:
                if response.status_code >= 400:
                    try:
                        response.read()
                    except Exception:
                        pass
                response.raise_for_status()
            except Exception as e:
                stream_ctx.__exit__(type(e), e, e.__traceback__)
                raise
            return stream_ctx, response

        completion_parts: List[str] = []
        thinking_parts: List[str] = []
        final_model = effective_model
        stop_reason: Optional[str] = None
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        stream_completed = False

        try:
            stream_ctx, response = self._execute_with_retry(operation=_open_stream)
            try:
                for data_str in iter_sse_data(response.iter_lines()):
                    data_str = data_str.strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        stream_completed = True
                        break

                    try:
                        chunk = json_loads(data_str)
                    except (ValueError, TypeError) as exc:
                        raise LLMError("Provider returned malformed stream event JSON") from exc
                    if not isinstance(chunk, dict):
                        raise LLMError("Provider stream event must be a JSON object")
                    if chunk.get("error"):
                        raise LLMError("Provider returned a streaming error")
                    if "choices" in chunk and not isinstance(chunk["choices"], list):
                        raise LLMError(
                            "Provider returned a malformed stream event: expected choices"
                        )
                    if "choices" not in chunk and not isinstance(chunk.get("usage"), dict):
                        raise LLMError(
                            "Provider returned a malformed stream event: expected choices or usage"
                        )

                    final_model = chunk.get("model", final_model)

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage_data["prompt_tokens"] = (
                            chunk_usage.get("prompt_tokens", usage_data["prompt_tokens"])
                            or usage_data["prompt_tokens"]
                        )
                        usage_data["completion_tokens"] = (
                            chunk_usage.get("completion_tokens", usage_data["completion_tokens"])
                            or usage_data["completion_tokens"]
                        )
                        usage_data["total_tokens"] = chunk_usage.get("total_tokens") or (
                            usage_data["prompt_tokens"] + usage_data["completion_tokens"]
                        )

                    for choice in chunk.get("choices", []) or []:
                        delta = choice.get("delta", {}) or {}

                        delta_thinking = self._extract_reasoning_text(delta)
                        if delta_thinking:
                            thinking_parts.append(delta_thinking)
                            yield LLMResponse(
                                model_id=final_model,
                                type="llm_delta",
                                text=None,
                                thinking=delta_thinking,
                                usage=None,
                                stop_reason=None,
                            )

                        delta_content = delta.get("content")
                        if isinstance(delta_content, str) and delta_content:
                            completion_parts.append(delta_content)
                            yield LLMResponse(
                                model_id=final_model,
                                type="llm_delta",
                                text=delta_content,
                                thinking=None,
                                usage=None,
                                stop_reason=None,
                            )
                        elif isinstance(delta_content, list):
                            text_fragments: List[str] = []
                            for part in delta_content:
                                if isinstance(part, dict) and part.get("type") in {
                                    "text",
                                    "output_text",
                                }:
                                    text_value = part.get("text")
                                    if text_value:
                                        text_fragments.append(text_value)
                            if text_fragments:
                                merged = "".join(text_fragments)
                                completion_parts.append(merged)
                                yield LLMResponse(
                                    model_id=final_model,
                                    type="llm_delta",
                                    text=merged,
                                    thinking=None,
                                    usage=None,
                                    stop_reason=None,
                                )

                        for tc in delta.get("tool_calls", []) or []:
                            idx = int(tc.get("index", len(tool_calls_by_index)))
                            entry = tool_calls_by_index.setdefault(
                                idx,
                                {"id": "", "function_name": "", "arguments_parts": []},
                            )
                            if tc.get("id"):
                                entry["id"] = tc["id"]

                            function = tc.get("function", {}) if isinstance(tc, dict) else {}
                            if function.get("name"):
                                entry["function_name"] = function["name"]
                            arguments_delta = function.get("arguments")
                            if isinstance(arguments_delta, str) and arguments_delta:
                                entry["arguments_parts"].append(arguments_delta)

                        if choice.get("finish_reason") is not None:
                            stop_reason = choice.get("finish_reason")
                            stream_completed = True
            finally:
                stream_ctx.__exit__(None, None, None)

            if not stream_completed:
                raise LLMError("Provider stream ended without a completion event")
            if tool_calls_by_index and stop_reason in {"length", "content_filter"}:
                raise LLMError("Provider response ended before tool calls completed")
            usage_obj = LLMUsage(**usage_data)
            aggregated_tool_calls: List[Dict[str, Any]] = []
            for idx in sorted(tool_calls_by_index.keys()):
                entry = tool_calls_by_index[idx]
                aggregated_tool_calls.append(
                    {
                        "id": entry["id"] or str(idx),
                        "function": {
                            "name": entry["function_name"],
                            "arguments": "".join(entry["arguments_parts"]),
                        },
                    }
                )

            if aggregated_tool_calls:
                tool_calls_obj = self._parse_tool_call_sync(aggregated_tool_calls)
                yield ToolCallList(model_id=final_model, tool_calls=tool_calls_obj, usage=usage_obj)
            else:
                yield LLMResponse(
                    model_id=final_model,
                    type="llm_response",
                    text="".join(completion_parts) if completion_parts else None,
                    thinking="".join(thinking_parts) if thinking_parts else None,
                    usage=usage_obj,
                    stop_reason=stop_reason,
                )
        except Exception as e:
            mapped_error = self._handle_error(e)
            raise mapped_error from safe_exception_cause(e, raised=mapped_error)
