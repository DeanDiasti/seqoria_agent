"""
Ollama provider implementation using direct HTTPX API calls.

This module provides synchronous and asynchronous provider classes for
interacting with Ollama's HTTP API (`/api/chat`, `/api/tags`, `/api/pull`),
supporting chat completions, streaming, tool calling, and vision.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import httpx

from seqoria_agent.client.ai_client.utils import json_loads
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
    QuotaExceededError,
    ServerError,
)
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList

_EMPTY_JSON: str = "{}"


class _OllamaProviderMixin:
    """Shared functionality between sync and async Ollama providers."""

    def _init_common(
        self,
        model_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> tuple:
        """Common initialization logic shared between sync and async providers."""
        if not model_id:
            raise InvalidRequestError(message="model_id is required for Ollama provider")

        effective_host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        effective_api_key = api_key or os.environ.get("OLLAMA_API_KEY")

        headers: Dict[str, str] = {}
        if effective_api_key:
            headers["Authorization"] = f"Bearer {effective_api_key}"

        max_retries = kwargs.get("max_retries", 3)
        auto_pull = kwargs.get("auto_pull", True)

        # Valid parameter keys accepted by Ollama /api/chat payload.
        valid_query_keys = {
            "format",
            "keep_alive",
            "think",
            "logprobs",
            "top_logprobs",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "num_ctx",
            "num_predict",
        }

        default_params: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value

        return effective_host, headers, model_id, default_params, max_retries, auto_pull

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters by merging defaults with overrides."""
        params = self.default_params.copy()
        params.update(kwargs)
        params = {k: v for k, v in params.items() if v is not None}

        if "keep_alive" not in params:
            keep_alive_value = getattr(self, "keep_alive", None)
            if keep_alive_value is not None:
                params["keep_alive"] = keep_alive_value

        return params

    def _build_options(self, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Split params into Ollama top-level keys and nested options keys."""
        top_level_keys = {"format", "keep_alive", "think", "logprobs", "top_logprobs"}
        options_keys = {"temperature", "top_p", "top_k", "min_p", "seed", "stop", "num_ctx", "num_predict"}

        top_level: Dict[str, Any] = {}
        options: Dict[str, Any] = {}

        for key, value in params.items():
            if key in top_level_keys:
                top_level[key] = value
            elif key in options_keys:
                options[key] = value

        return top_level, options

    def _extract_error_details(
        self, error: httpx.HTTPStatusError
    ) -> tuple[int, str, Optional[str], Optional[str]]:
        """Extract normalized status/message/type/param from Ollama error payloads."""
        status_code = error.response.status_code
        message = str(error)
        code: Optional[str] = None
        param: Optional[str] = None

        try:
            payload = error.response.json()
            if isinstance(payload, dict):
                if payload.get("error") is not None:
                    message = str(payload["error"])
                elif payload.get("message") is not None:
                    message = str(payload["message"])
                raw_code = payload.get("code")
                if raw_code is not None:
                    code = str(raw_code)
                raw_param = payload.get("param")
                if raw_param is not None:
                    param = str(raw_param)
        except Exception:
            pass

        return status_code, message, code, param

    def _handle_error(self, error: Exception) -> Exception:
        """Map Ollama HTTP errors to standardized provider errors."""
        if isinstance(error, LLMError):
            return error

        if isinstance(error, httpx.HTTPStatusError):
            status_code, message, _code, param = self._extract_error_details(error)
            message_l = message.lower()

            if status_code == 401:
                return EnvironmentError(
                    message=f"Ollama authentication failed: {message}",
                    missing_vars=["OLLAMA_API_KEY"],
                )
            if status_code == 404:
                if "model" in message_l or "not found" in message_l:
                    return ModelNotFoundError(model_id=self.model_id, provider="Ollama")
                return InvalidRequestError(message=message, param=param)
            if status_code == 429:
                return QuotaExceededError(provider="Ollama", limit_type="rate")
            if status_code == 400:
                if "context length" in message_l or "too long" in message_l:
                    return ContextLengthExceededError(model_id=self.model_id, tokens=None, max_tokens=None)
                return InvalidRequestError(message=message, param=param)
            if status_code == 403:
                return InvalidRequestError(message=f"Ollama permission denied: {message}", param=param)
            if 500 <= status_code < 600:
                return ServerError(provider="Ollama", status_code=status_code)
            return LLMError(message=f"Ollama API Error {status_code}: {message}")

        if isinstance(error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return ConnectionError(provider="Ollama", message="Request timed out")
        if isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).lower():
            return ConnectionError(provider="Ollama", message="Request timed out")
        if isinstance(error, (httpx.NetworkError, httpx.TransportError, httpx.ConnectError)):
            return ConnectionError(provider="Ollama", message=f"Connection error: {error}")
        if isinstance(error, httpx.HTTPError):
            return ConnectionError(provider="Ollama", message=f"HTTP error: {error}")

        error_str = str(error).lower()
        if "connection" in error_str or "refused" in error_str or "unreachable" in error_str:
            return ConnectionError(provider="Ollama", message=f"Connection error: {error}")
        if "timeout" in error_str or "timed out" in error_str:
            return ConnectionError(provider="Ollama", message="Request timed out")
        if "context length" in error_str or "too long" in error_str:
            return ContextLengthExceededError(model_id=self.model_id, tokens=None, max_tokens=None)
        if "not found" in error_str:
            return ModelNotFoundError(model_id=self.model_id, provider="Ollama")

        return LLMError(message=f"An unexpected Ollama error occurred: {error}")

    def _convert_image_content_ollama(self, content: Union[str, List[Dict[str, Any]]]) -> tuple[str, List[str]]:
        """Convert common multimodal blocks to Ollama's text + images format."""
        if isinstance(content, str):
            return content, []

        text_parts: List[str] = []
        images: List[str] = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
                continue

            if block_type == "image":
                if "base64" in block:
                    images.append(block["base64"])
                elif "url" in block:
                    url = block["url"]
                    if url.startswith("data:"):
                        try:
                            images.append(url.split(",", 1)[1])
                        except (IndexError, ValueError):
                            pass
                    else:
                        text_parts.append(f"[Image URL: {url}]")
                continue

            if block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                if url.startswith("data:"):
                    try:
                        images.append(url.split(",", 1)[1])
                    except (IndexError, ValueError):
                        pass
                else:
                    text_parts.append(f"[Image URL: {url}]")
                continue

        return " ".join(text_parts), images

    def _prepare_chat_history_sync(
        self,
        chat_history: List[
            Union[Dict[str, Any], ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage]
        ],
    ) -> List[Dict[str, Any]]:
        """Prepare chat history for Ollama API."""
        prepared_messages: List[Dict[str, Any]] = []

        for raw_msg in chat_history:
            if isinstance(raw_msg, ToolCallRequest):
                prepared_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": raw_msg.name,
                                    "arguments": raw_msg.arguments or {},
                                }
                            }
                        ],
                    }
                )
                continue

            if isinstance(raw_msg, ToolCallResult):
                prepared_messages.append({"role": "tool", "content": str(raw_msg.result)})
                continue

            if isinstance(raw_msg, (AssistantMessage, UserMessage, SystemMessage)):
                text_content, images = self._convert_image_content_ollama(raw_msg.content)
                msg: Dict[str, Any] = {"role": raw_msg.role, "content": text_content}
                if images:
                    msg["images"] = images
                prepared_messages.append(msg)
                continue

            role = raw_msg.get("role")
            content = raw_msg.get("content", "")

            if role == "tool_call":
                prepared_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": raw_msg.get("name"),
                                    "arguments": raw_msg.get("arguments", {}),
                                }
                            }
                        ],
                    }
                )
                continue

            if role in {"tool_result", "tool"}:
                prepared_messages.append({"role": "tool", "content": str(raw_msg.get("result", content))})
                continue

            if role in {"assistant", "system", "user"}:
                text_content, images = self._convert_image_content_ollama(content)
                msg = {"role": role, "content": text_content}
                if images:
                    msg["images"] = images

                tool_calls = raw_msg.get("tool_calls")
                if role == "assistant" and isinstance(tool_calls, list):
                    msg["tool_calls"] = tool_calls

                prepared_messages.append(msg)

        return prepared_messages

    async def _prepare_chat_history_async(
        self,
        chat_history: List[
            Union[Dict[str, Any], ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage]
        ],
    ) -> List[Dict[str, Any]]:
        """Async wrapper around sync chat-history preparation."""
        return self._prepare_chat_history_sync(chat_history)

    def _convert_tools_for_ollama(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to Ollama's function-tool schema."""
        if not tools:
            return None

        ollama_tools: List[Dict[str, Any]] = []
        for tool in tools:
            function_block = tool.get("function")
            if isinstance(function_block, dict):
                name = function_block.get("name") or tool.get("name")
                description = function_block.get("description") or tool.get("description", "") or ""
                parameters = (
                    function_block.get("parameters")
                    or tool.get("input_schema")
                    or tool.get("parameters", {})
                    or {}
                )
            else:
                name = tool.get("name")
                description = tool.get("description", "") or ""
                parameters = tool.get("input_schema", tool.get("parameters", {})) or {}

            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )

        return ollama_tools

    def _parse_tool_calls(self, tool_calls: Optional[List[Dict[str, Any]]]) -> List[ToolCall]:
        """Parse tool calls from Ollama response payload."""
        if not tool_calls:
            return []

        parsed: List[ToolCall] = []
        for i, tc in enumerate(tool_calls):
            function_info = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = function_info.get("arguments", {})
            if isinstance(args, str):
                try:
                    decoded = json_loads(args)
                    args = decoded if isinstance(decoded, dict) else {}
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            parsed.append(
                ToolCall(
                    id=str(tc.get("id", i)) if isinstance(tc, dict) else str(i),
                    type="function_call",
                    function_name=str(function_info.get("name", "")),
                    arguments=args,
                )
            )

        return parsed

    def _iter_ndjson_events(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Yield parsed NDJSON events from sync streaming response."""
        for line in response.iter_lines():
            if line is None:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json_loads(stripped)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload

    async def _aiter_ndjson_events(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield parsed NDJSON events from async streaming response."""
        async for line in response.aiter_lines():
            if line is None:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json_loads(stripped)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload

    @staticmethod
    def _extract_model_names_from_tags_payload(payload: Dict[str, Any]) -> List[str]:
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        names: List[str] = []
        seen: set[str] = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model", "remote_model"):
                raw = item.get(key)
                if raw:
                    name = str(raw)
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
        return names

    def _validate_model_sync(self, client: Any, model_id: str, auto_pull: bool = True) -> None:  # noqa: ANN401
        """Validate that model is available, with optional auto-pull."""
        try:
            # Legacy SDK compatibility path used by tests and external callers.
            if hasattr(client, "list") and callable(client.list):
                response = client.list()
                models = response.get("models", []) if isinstance(response, dict) else []
                full_model_names = [m.get("name", "") for m in models if isinstance(m, dict)]
                available_model_names = [n.split(":")[0] for n in full_model_names]

                model_base = model_id.split(":")[0]
                if model_id in full_model_names or model_base in available_model_names:
                    return

                if auto_pull:
                    try:
                        if hasattr(client, "pull") and callable(client.pull):
                            for _ in client.pull(model_id, stream=True):
                                pass
                            return
                    except Exception as pull_error:
                        raise ModelNotFoundError(
                            model_id=model_id,
                            provider="Ollama",
                            available_models=full_model_names,
                        ) from pull_error

                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Ollama",
                    available_models=full_model_names,
                )

            # HTTPX path for migrated provider.
            tags_resp = client.get("/api/tags")
            tags_resp.raise_for_status()
            tags_payload = tags_resp.json()
            if not isinstance(tags_payload, dict):
                tags_payload = {}

            full_model_names = self._extract_model_names_from_tags_payload(tags_payload)
            available_model_names = [n.split(":")[0] for n in full_model_names]
            model_base = model_id.split(":")[0]

            if model_id in full_model_names or model_base in available_model_names:
                return

            if auto_pull:
                stream_ctx = client.stream("POST", "/api/pull", json={"model": model_id, "stream": True})
                response = stream_ctx.__enter__()
                try:
                    response.raise_for_status()
                    for event in self._iter_ndjson_events(response):
                        if event.get("error"):
                            raise ModelNotFoundError(
                                model_id=model_id,
                                provider="Ollama",
                                available_models=full_model_names,
                            )
                finally:
                    stream_ctx.__exit__(None, None, None)
                return

            raise ModelNotFoundError(
                model_id=model_id,
                provider="Ollama",
                available_models=full_model_names,
            )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)

    async def _validate_model_async(self, client: Any, model_id: str, auto_pull: bool = True) -> None:  # noqa: ANN401
        """Async model validation with optional auto-pull."""
        try:
            # Legacy SDK compatibility path used by tests and external callers.
            if hasattr(client, "list") and callable(client.list):
                response = await client.list()
                models = response.get("models", []) if isinstance(response, dict) else []
                full_model_names = [m.get("name", "") for m in models if isinstance(m, dict)]
                available_model_names = [n.split(":")[0] for n in full_model_names]

                model_base = model_id.split(":")[0]
                if model_id in full_model_names or model_base in available_model_names:
                    return

                if auto_pull:
                    try:
                        if hasattr(client, "pull") and callable(client.pull):
                            async for _ in await client.pull(model_id, stream=True):
                                pass
                            return
                    except Exception as pull_error:
                        raise ModelNotFoundError(
                            model_id=model_id,
                            provider="Ollama",
                            available_models=full_model_names,
                        ) from pull_error

                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Ollama",
                    available_models=full_model_names,
                )

            # HTTPX path for migrated provider.
            tags_resp = await client.get("/api/tags")
            tags_resp.raise_for_status()
            tags_payload = tags_resp.json()
            if not isinstance(tags_payload, dict):
                tags_payload = {}

            full_model_names = self._extract_model_names_from_tags_payload(tags_payload)
            available_model_names = [n.split(":")[0] for n in full_model_names]
            model_base = model_id.split(":")[0]

            if model_id in full_model_names or model_base in available_model_names:
                return

            if auto_pull:
                stream_ctx = client.stream("POST", "/api/pull", json={"model": model_id, "stream": True})
                response = await stream_ctx.__aenter__()
                try:
                    response.raise_for_status()
                    async for event in self._aiter_ndjson_events(response):
                        if event.get("error"):
                            raise ModelNotFoundError(
                                model_id=model_id,
                                provider="Ollama",
                                available_models=full_model_names,
                            )
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                return

            raise ModelNotFoundError(
                model_id=model_id,
                provider="Ollama",
                available_models=full_model_names,
            )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)


class AsyncOllamaProvider(AsyncBaseProvider, _OllamaProviderMixin):
    """Asynchronous provider implementation for Ollama HTTP API."""

    def __init__(
        self,
        model_id: str,
        host: Optional[str] = "http://localhost:11434",
        api_key: Optional[str] = None,
        keep_alive: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.keep_alive = keep_alive
        timeout = kwargs.get("timeout", 30.0)

        (
            self.host,
            self.headers,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.auto_pull,
        ) = self._init_common(model_id, host, api_key, **kwargs)

        try:
            self.async_client = httpx.AsyncClient(
                base_url=self.host,
                headers=self.headers,
                timeout=timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="Ollama", message=str(e)) from e

    async def validate_model(self) -> None:
        """Validate that configured model is available."""
        await self._validate_model_async(self.async_client, self.model_id, self.auto_pull)

    async def async_generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Asynchronously generate a completion for provided chat history."""
        del tool_choice  # Ollama does not currently support OpenAI-style tool_choice semantics.

        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        messages = await self._prepare_chat_history_async(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            **top_level,
        }
        if options:
            payload["options"] = options
        if ollama_tools:
            payload["tools"] = ollama_tools

        async def _call():
            response = await self.async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise LLMError(message="Ollama API returned an unexpected response payload.")
            return data

        try:
            response = await self._execute_with_retry(_call)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise

        message = response.get("message", {}) if isinstance(response.get("message"), dict) else {}
        content = message.get("content", "")
        thinking = message.get("thinking")
        tool_calls = message.get("tool_calls")

        usage = None
        prompt_eval_count = response.get("prompt_eval_count")
        eval_count = response.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            usage = LLMUsage(
                prompt_tokens=prompt_eval_count or 0,
                completion_tokens=eval_count or 0,
                total_tokens=(prompt_eval_count or 0) + (eval_count or 0),
            )

        if tool_calls:
            return ToolCallList(
                model_id=response.get("model", effective_model),
                tool_calls=self._parse_tool_calls(tool_calls),
                usage=usage,
            )

        return LLMResponse(
            model_id=response.get("model", effective_model),
            type="llm_response",
            text=content,
            thinking=thinking,
            usage=usage,
            stop_reason=response.get("done_reason"),
        )

    async def async_stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """Asynchronously stream a completion for provided chat history."""
        del tool_choice  # Ollama does not currently support OpenAI-style tool_choice semantics.

        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        messages = await self._prepare_chat_history_async(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
            **top_level,
        }
        if options:
            payload["options"] = options
        if ollama_tools:
            payload["tools"] = ollama_tools

        async def _open_stream():
            stream_ctx = self.async_client.stream("POST", "/api/chat", json=payload)
            response = await stream_ctx.__aenter__()
            try:
                response.raise_for_status()
            except Exception as e:
                await stream_ctx.__aexit__(type(e), e, e.__traceback__)
                raise
            return stream_ctx, response

        full_content = ""
        full_thinking = ""
        final_model = effective_model
        final_usage = None
        stop_reason = None
        tool_calls_accumulated: List[Dict[str, Any]] = []

        try:
            stream_ctx, response = await self._execute_with_retry(_open_stream)
            try:
                async for chunk in self._aiter_ndjson_events(response):
                    message = chunk.get("message", {}) if isinstance(chunk.get("message"), dict) else {}
                    content_delta = message.get("content", "")
                    thinking_delta = message.get("thinking", "")

                    if message.get("tool_calls"):
                        tool_calls_raw = message.get("tool_calls")
                        if isinstance(tool_calls_raw, list):
                            tool_calls_accumulated.extend(tool_calls_raw)

                    if thinking_delta:
                        full_thinking += thinking_delta
                        yield LLMResponse(
                            model_id=chunk.get("model", effective_model),
                            type="llm_delta",
                            text=None,
                            thinking=thinking_delta,
                            usage=None,
                            stop_reason=None,
                        )

                    if content_delta:
                        full_content += content_delta
                        yield LLMResponse(
                            model_id=chunk.get("model", effective_model),
                            type="llm_delta",
                            text=content_delta,
                            thinking=None,
                            usage=None,
                            stop_reason=None,
                        )

                    if chunk.get("done"):
                        final_model = chunk.get("model", effective_model)
                        stop_reason = chunk.get("done_reason")
                        prompt_eval_count = chunk.get("prompt_eval_count")
                        eval_count = chunk.get("eval_count")
                        if prompt_eval_count is not None or eval_count is not None:
                            final_usage = LLMUsage(
                                prompt_tokens=prompt_eval_count or 0,
                                completion_tokens=eval_count or 0,
                                total_tokens=(prompt_eval_count or 0) + (eval_count or 0),
                            )
            finally:
                await stream_ctx.__aexit__(None, None, None)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise

        if tool_calls_accumulated:
            yield ToolCallList(
                model_id=final_model,
                tool_calls=self._parse_tool_calls(tool_calls_accumulated),
                usage=final_usage,
            )
        else:
            yield LLMResponse(
                model_id=final_model,
                type="llm_response",
                text=full_content,
                thinking=full_thinking if full_thinking else None,
                usage=final_usage,
                stop_reason=stop_reason,
            )


class OllamaProvider(BaseProvider, _OllamaProviderMixin):
    """Synchronous provider implementation for Ollama HTTP API."""

    def __init__(
        self,
        model_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        validate_on_init: bool = False,
        keep_alive: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.keep_alive = keep_alive
        timeout = kwargs.get("timeout", 30.0)

        (
            self.host,
            self.headers,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.auto_pull,
        ) = self._init_common(model_id, host, api_key, **kwargs)

        try:
            self.client = httpx.Client(
                base_url=self.host,
                headers=self.headers,
                timeout=timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="Ollama", message=str(e)) from e

        if validate_on_init:
            self.validate_model()

    def validate_model(self) -> None:
        """Validate that configured model is available."""
        self._validate_model_sync(self.client, self.model_id, self.auto_pull)

    def generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Synchronously generate a completion for provided chat history."""
        del tool_choice  # Ollama does not currently support OpenAI-style tool_choice semantics.

        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        messages = self._prepare_chat_history_sync(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            **top_level,
        }
        if options:
            payload["options"] = options
        if ollama_tools:
            payload["tools"] = ollama_tools

        def _call():
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise LLMError(message="Ollama API returned an unexpected response payload.")
            return data

        try:
            response = self._execute_with_retry(_call)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise

        message = response.get("message", {}) if isinstance(response.get("message"), dict) else {}
        content = message.get("content", "")
        thinking = message.get("thinking")
        tool_calls = message.get("tool_calls")

        usage = None
        prompt_eval_count = response.get("prompt_eval_count")
        eval_count = response.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            usage = LLMUsage(
                prompt_tokens=prompt_eval_count or 0,
                completion_tokens=eval_count or 0,
                total_tokens=(prompt_eval_count or 0) + (eval_count or 0),
            )

        if tool_calls:
            return ToolCallList(
                model_id=response.get("model", effective_model),
                tool_calls=self._parse_tool_calls(tool_calls),
                usage=usage,
            )

        return LLMResponse(
            model_id=response.get("model", effective_model),
            type="llm_response",
            text=content,
            thinking=thinking,
            usage=usage,
            stop_reason=response.get("done_reason"),
        )

    def stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        """Synchronously stream a completion for provided chat history."""
        del tool_choice  # Ollama does not currently support OpenAI-style tool_choice semantics.

        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        messages = self._prepare_chat_history_sync(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
            **top_level,
        }
        if options:
            payload["options"] = options
        if ollama_tools:
            payload["tools"] = ollama_tools

        def _open_stream():
            stream_ctx = self.client.stream("POST", "/api/chat", json=payload)
            response = stream_ctx.__enter__()
            try:
                response.raise_for_status()
            except Exception as e:
                stream_ctx.__exit__(type(e), e, e.__traceback__)
                raise
            return stream_ctx, response

        full_content = ""
        full_thinking = ""
        final_model = effective_model
        final_usage = None
        stop_reason = None
        tool_calls_accumulated: List[Dict[str, Any]] = []

        try:
            stream_ctx, response = self._execute_with_retry(_open_stream)
            try:
                for chunk in self._iter_ndjson_events(response):
                    message = chunk.get("message", {}) if isinstance(chunk.get("message"), dict) else {}
                    content_delta = message.get("content", "")
                    thinking_delta = message.get("thinking", "")

                    if message.get("tool_calls"):
                        tool_calls_raw = message.get("tool_calls")
                        if isinstance(tool_calls_raw, list):
                            tool_calls_accumulated.extend(tool_calls_raw)

                    if thinking_delta:
                        full_thinking += thinking_delta
                        yield LLMResponse(
                            model_id=chunk.get("model", effective_model),
                            type="llm_delta",
                            text=None,
                            thinking=thinking_delta,
                            usage=None,
                            stop_reason=None,
                        )

                    if content_delta:
                        full_content += content_delta
                        yield LLMResponse(
                            model_id=chunk.get("model", effective_model),
                            type="llm_delta",
                            text=content_delta,
                            thinking=None,
                            usage=None,
                            stop_reason=None,
                        )

                    if chunk.get("done"):
                        final_model = chunk.get("model", effective_model)
                        stop_reason = chunk.get("done_reason")
                        prompt_eval_count = chunk.get("prompt_eval_count")
                        eval_count = chunk.get("eval_count")
                        if prompt_eval_count is not None or eval_count is not None:
                            final_usage = LLMUsage(
                                prompt_tokens=prompt_eval_count or 0,
                                completion_tokens=eval_count or 0,
                                total_tokens=(prompt_eval_count or 0) + (eval_count or 0),
                            )
            finally:
                stream_ctx.__exit__(None, None, None)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise

        if tool_calls_accumulated:
            yield ToolCallList(
                model_id=final_model,
                tool_calls=self._parse_tool_calls(tool_calls_accumulated),
                usage=final_usage,
            )
        else:
            yield LLMResponse(
                model_id=final_model,
                type="llm_response",
                text=full_content,
                thinking=full_thinking if full_thinking else None,
                usage=final_usage,
                stop_reason=stop_reason,
            )

