"""Anthropic provider implementation using HTTPX + Messages API."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import httpx

from seqoria_agent.client.ai_client.utils import json_loads
from seqoria_agent.models.chunk import (
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolCallResult,
    ToolCallError,
    UserMessage,
)

from .base_provider import AsyncBaseProvider, BaseProvider
from .errors import (
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
from .schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList

_EMPTY_JSON: str = "{}"


class _AnthropicProviderMixin:
    """Shared functionality between sync and async Anthropic providers."""

    def _init_common(self, model_id: str, api_key: Optional[str], **kwargs) -> tuple:
        """Common initialization logic shared between sync and async providers."""
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise EnvironmentError(message="Anthropic API key not provided", missing_vars=["ANTHROPIC_API_KEY"])

        if not model_id:
            raise InvalidRequestError(message="model_id is required for Anthropic provider")

        max_retries = kwargs.get("max_retries", 3)
        timeout = kwargs.get("timeout", 30.0)
        base_url = (kwargs.get("base_url") or "https://api.anthropic.com").rstrip("/")
        anthropic_version = kwargs.get("anthropic_version", "2023-06-01")
        anthropic_beta = kwargs.get("anthropic_beta")

        headers: Dict[str, str] = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "Content-Type": "application/json",
        }
        if anthropic_beta:
            if isinstance(anthropic_beta, (list, tuple, set)):
                headers["anthropic-beta"] = ",".join(str(v) for v in anthropic_beta if v is not None)
            else:
                headers["anthropic-beta"] = str(anthropic_beta)

        client_kwargs = {
            "base_url": base_url,
            "headers": headers,
            "timeout": timeout,
            "http2": True,
        }

        # Valid Anthropic /v1/messages request params, excluding model/messages/system/stream.
        valid_query_keys = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "thinking",
            "service_tier",
            "metadata",
            "inference_geo",
            "output_config",
        }

        default_params: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value

        return api_key, client_kwargs, model_id, default_params, max_retries

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters by merging defaults with call overrides."""
        params = self.default_params.copy()
        params.update(kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def _safe_response_text(self, response: httpx.Response, *, max_chars: int = 1200) -> Optional[str]:
        """Best-effort extraction of response body text for error transparency."""
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

    def _extract_error_details(
        self, error: httpx.HTTPStatusError
    ) -> tuple[int, str, Optional[str], Optional[str]]:
        """Extract normalized status/message/type/param from Anthropic error payloads."""
        response = error.response
        status_code = response.status_code
        message = str(error)
        error_type: Optional[str] = None
        param: Optional[str] = None

        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    message = str(err.get("message", message))
                    error_type = str(err.get("type", "")) or None
                    p = err.get("param")
                    param = str(p) if p is not None else None
                elif isinstance(err, str) and err.strip():
                    message = err
                elif payload.get("message"):
                    message = str(payload["message"])
        except Exception:
            pass

        if message == str(error) or message.lower().startswith(("client error", "server error")):
            raw_body = self._safe_response_text(response)
            if raw_body:
                message = raw_body

        request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
        if request_id and f"request_id={request_id}" not in message:
            message = f"{message} (request_id={request_id})"

        return status_code, message, error_type, param

    def _handle_error(self, error: Exception) -> Exception:
        """Map HTTPX/Anthropic API errors into provider-standard errors."""
        if isinstance(error, LLMError):
            return error

        if isinstance(error, httpx.HTTPStatusError):
            status_code, message, error_type, param = self._extract_error_details(error)
            message_l = message.lower()
            error_type_l = (error_type or "").lower()

            if status_code == 401 or error_type_l == "authentication_error":
                return EnvironmentError(
                    message=f"Anthropic authentication failed: {message}",
                    missing_vars=["ANTHROPIC_API_KEY"],
                )

            if status_code == 429 or error_type_l == "rate_limit_error":
                return QuotaExceededError(provider="Anthropic", limit_type="rate")

            if status_code == 404 or error_type_l == "not_found_error":
                if "model" in message_l or "model" in error_type_l:
                    return ModelNotFoundError(model_id=self.model_id, provider="Anthropic")
                return InvalidRequestError(message=message, param=param)

            if status_code == 400:
                if "context_length_exceeded" in message_l or "maximum context length" in message_l:
                    numbers = [int(n) for n in re.findall(r"\d+", message)]
                    tokens = numbers[0] if numbers else None
                    max_tokens = numbers[1] if len(numbers) > 1 else None
                    return ContextLengthExceededError(
                        model_id=self.model_id,
                        tokens=tokens,
                        max_tokens=max_tokens,
                    )
                if "content_policy_violation" in message_l or "safety" in message_l:
                    return ContentFilterError(provider="Anthropic", filter_reason=message)
                if "tool_choice" in message_l:
                    return InvalidRequestError(message=f"Tool choice configuration error: {message}", param=param)
                if "input_schema" in message_l:
                    return InvalidRequestError(message=f"Tool schema validation error: {message}", param=param)
                if "tool" in message_l and ("invalid" in message_l or "missing" in message_l):
                    return InvalidRequestError(message=f"Tool validation error: {message}", param=param)
                return InvalidRequestError(message=message, param=param)

            if status_code == 403 or error_type_l == "permission_error":
                return InvalidRequestError(message=f"Anthropic permission denied: {message}", param=param)

            if status_code == 422:
                return ContentFilterError(provider="Anthropic", filter_reason=message)

            if status_code >= 500 or error_type_l in {"overloaded_error", "api_error"}:
                return ServerError(provider="Anthropic", status_code=status_code)

            return LLMError(message=f"Anthropic API Error {status_code}: {message}")

        if isinstance(error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return ConnectionError(provider="Anthropic", message="Request timed out")
        if isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).lower():
            return ConnectionError(provider="Anthropic", message="Request timed out")
        if isinstance(error, (httpx.NetworkError, httpx.TransportError, httpx.ConnectError)):
            return ConnectionError(provider="Anthropic", message=f"Connection error: {error}")
        if isinstance(error, httpx.HTTPError):
            return ConnectionError(provider="Anthropic", message=f"HTTP error: {error}")

        return LLMError(message=f"An unexpected Anthropic error occurred: {error}")

    def _parse_tool_call_sync(self, tool_call: List[Any]) -> List[ToolCall]:
        """Convert raw tool-call payloads into ToolCall schema objects."""
        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            if isinstance(tc, ToolCall):
                processed_tool_calls.append(tc)
                continue

            if isinstance(tc, dict):
                args = tc.get("arguments", tc.get("input", {}))
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except Exception:
                        args = {}

                processed_tool_calls.append(
                    ToolCall(
                        id=str(tc.get("id", "")),
                        type="function_call",
                        function_name=str(tc.get("function_name", tc.get("name", ""))),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
                continue

            args = getattr(tc, "arguments", getattr(tc, "input", {}))
            if isinstance(args, str):
                try:
                    args = json_loads(args)
                except Exception:
                    args = {}

            processed_tool_calls.append(
                ToolCall(
                    id=str(getattr(tc, "id", "")),
                    type="function_call",
                    function_name=str(getattr(tc, "function_name", getattr(tc, "name", ""))),
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        return processed_tool_calls

    def _format_tool_choice(self, tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format tool_choice for Anthropic Messages API."""
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "any", "none"}:
                return {"type": tool_choice}
            return {"type": "tool", "name": tool_choice}

        if isinstance(tool_choice, dict):
            if "type" not in tool_choice:
                if "name" in tool_choice:
                    return {"type": "tool", **tool_choice}
                return {"type": "auto", **tool_choice}
            return tool_choice

        return None

    def _format_tool_call_sync(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools to the schema expected by Anthropic Messages API."""
        formatted_tools: List[Dict[str, Any]] = []

        for tool in tools:
            if "type" in tool and tool["type"] in {
                "bash_20250124",
                "text_editor_20250124",
                "text_editor_20250429",
                "text_editor_20250728",
                "web_search_20250305",
            }:
                formatted_tools.append(tool)
                continue

            if "function" in tool:
                func = tool["function"]
                formatted_tool: Dict[str, Any] = {
                    "name": str(func.get("name", "")),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
                if "cache_control" in tool:
                    formatted_tool["cache_control"] = tool["cache_control"]
                if "strict" in func:
                    formatted_tool["strict"] = func["strict"]
                formatted_tools.append(formatted_tool)
                continue

            if "name" in tool and "input_schema" in tool:
                formatted_tools.append(tool)
                continue

            if "name" in tool:
                formatted_tools.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get(
                            "input_schema",
                            tool.get("parameters", {"type": "object", "properties": {}}),
                        ),
                    }
                )

        return formatted_tools

    def _convert_image_content_anthropic(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert image blocks into Anthropic input format."""
        converted: List[Dict[str, Any]] = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                converted.append({"type": "text", "text": block.get("text", "")})
                continue

            if block_type == "image":
                if "source" in block:
                    converted.append(block)
                elif "url" in block:
                    converted.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": block["url"]},
                        }
                    )
                elif "base64" in block:
                    converted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.get("media_type", "image/jpeg"),
                                "data": block["base64"],
                            },
                        }
                    )
                continue

            if block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                if url.startswith("data:"):
                    try:
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        converted.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    except (ValueError, IndexError):
                        converted.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            }
                        )
                else:
                    converted.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        }
                    )
                continue

            converted.append(block)

        return converted

    def _prepare_chat_history_sync(
        self,
        chat_history: List[Dict[str, Any]],
        *,
        enable_cache: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare system/message payloads for Anthropic Messages API."""
        prepared_chat_history: List[Dict[str, Any]] = []
        system_message: List[Dict[str, Any]] = []

        for message in chat_history:
            if isinstance(message, ToolCallRequest):
                args = message.arguments
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except (ValueError, TypeError):
                        args = {}
                elif args is None:
                    args = {}

                prepared_chat_history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": str(message.id),
                                "name": message.name,
                                "input": args if isinstance(args, dict) else {},
                            }
                        ],
                    }
                )
                continue

            if isinstance(message, (ToolCallResult, ToolCallError)):
                result_content = message.result
                content_value = result_content if isinstance(result_content, str) else str(result_content)

                prepared_chat_history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(message.tool_call_id),
                                "content": content_value,
                                **({"is_error": True} if isinstance(message, ToolCallError) else {}),
                            }
                        ],
                    }
                )
                continue

            if isinstance(message, (AssistantMessage, UserMessage)):
                content = message.content
                if isinstance(content, list):
                    content = self._convert_image_content_anthropic(content)
                    prepared_chat_history.append({"role": message.role, "content": content})
                else:
                    prepared_chat_history.append(
                        {
                            "role": message.role,
                            "content": [{"type": "text", "text": str(content) if content is not None else ""}],
                        }
                    )
                continue

            if isinstance(message, SystemMessage):
                sys_block = {"type": "text", "text": message.content}
                if enable_cache:
                    sys_block["cache_control"] = {"type": "ephemeral"}
                system_message.append(sys_block)
                continue

            role = message.get("role")

            if role == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    sys_block = {"type": "text", "text": content}
                    if enable_cache:
                        sys_block["cache_control"] = {"type": "ephemeral"}
                    system_message.append(sys_block)
                elif isinstance(content, list):
                    for block in content:
                        sys_block = block.copy()
                        if enable_cache:
                            sys_block["cache_control"] = {"type": "ephemeral"}
                        system_message.append(sys_block)
                else:
                    sys_block = {"type": "text", "text": str(content)}
                    if enable_cache:
                        sys_block["cache_control"] = {"type": "ephemeral"}
                    system_message.append(sys_block)
                continue

            if role == "tool_call":
                args = message.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except (ValueError, TypeError):
                        args = {}

                prepared_chat_history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": str(message.get("id", "")),
                                "name": message.get("name", ""),
                                "input": args if isinstance(args, dict) else {},
                            }
                        ],
                    }
                )
                continue

            if role in {"tool_result", "tool_error", "tool"}:
                result = (message.get("error", "") if role == "tool_error" else message.get("result", message.get("content", "")))
                tool_use_id = message.get("tool_call_id", message.get("tool_use_id", ""))
                is_error = role == "tool_error" or bool(message.get("is_error", False))

                tool_result_block: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": str(tool_use_id),
                    "content": result if isinstance(result, str) else str(result),
                }
                if is_error:
                    tool_result_block["is_error"] = True

                prepared_chat_history.append({"role": "user", "content": [tool_result_block]})
                continue

            if role in {"assistant", "user"}:
                content = message.get("content")
                if isinstance(content, list):
                    processed_content: List[Dict[str, Any]] = []
                    for block in content:
                        block_type = block.get("type")

                        if block_type == "thinking":
                            processed_content.append(
                                {
                                    "type": "thinking",
                                    "thinking": block.get("thinking", ""),
                                    "signature": block.get("signature", ""),
                                }
                            )
                            continue

                        if block_type == "redacted_thinking":
                            processed_content.append({"type": "redacted_thinking", "data": block.get("data", "")})
                            continue

                        if block_type == "tool_use":
                            args = block.get("input", {})
                            if isinstance(args, str):
                                try:
                                    args = json_loads(args)
                                except (ValueError, TypeError):
                                    args = {}
                            processed_content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "input": args if isinstance(args, dict) else {},
                                }
                            )
                            continue

                        if block_type == "tool_result":
                            tool_result_block = {
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            }
                            if block.get("is_error"):
                                tool_result_block["is_error"] = True
                            processed_content.append(tool_result_block)
                            continue

                        converted = self._convert_image_content_anthropic([block])
                        processed_content.extend(converted)

                    prepared_chat_history.append({"role": role, "content": processed_content})
                else:
                    prepared_chat_history.append(
                        {
                            "role": role,
                            "content": [{"type": "text", "text": str(content) if content is not None else ""}],
                        }
                    )

        if enable_cache and not system_message and prepared_chat_history:
            first_msg = prepared_chat_history[0]
            content = first_msg.get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}

        return system_message, prepared_chat_history

    def _extract_usage(self, payload: Dict[str, Any]) -> LLMUsage:
        """Extract normalized token usage from Anthropic response/event payloads."""
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _extract_message_output(
        self, payload: Dict[str, Any], fallback_model: str
    ) -> tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[str], str, LLMUsage]:
        """Extract text/thinking/tool-calls + metadata from a message payload."""
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        content_blocks = payload.get("content") or []
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type == "text":
                    txt = block.get("text")
                    if txt:
                        text_parts.append(txt)
                    continue

                if block_type == "thinking":
                    thinking = block.get("thinking")
                    if thinking:
                        thinking_parts.append(thinking)
                    continue

                if block_type == "tool_use":
                    args = block.get("input", {})
                    if isinstance(args, str):
                        try:
                            args = json_loads(args)
                        except Exception:
                            args = {}
                    tool_calls.append(
                        {
                            "id": str(block.get("id", "")),
                            "name": str(block.get("name", "")),
                            "function_name": str(block.get("name", "")),
                            "arguments": args if isinstance(args, dict) else {},
                        }
                    )

        model_used = str(payload.get("model", fallback_model))
        stop_reason = payload.get("stop_reason")
        usage_obj = self._extract_usage(payload)
        text = "".join(text_parts) if text_parts else None
        thinking = "".join(thinking_parts) if thinking_parts else None
        return text, thinking, tool_calls, stop_reason, model_used, usage_obj

    @staticmethod
    def _extract_model_ids_from_items(items: Any) -> List[str]:  # noqa: ANN401
        """Extract model IDs from mixed dict/object collections."""
        if not isinstance(items, list):
            return []

        ids: List[str] = []
        for item in items:
            model_id: Optional[str] = None
            if isinstance(item, dict):
                raw = item.get("id")
                if raw:
                    model_id = str(raw)
            else:
                raw = getattr(item, "id", None)
                if raw:
                    model_id = str(raw)

            if model_id:
                ids.append(model_id)
        return ids

    def _validate_model_sync(self, client: httpx.Client, model_id: str) -> None:
        """Validate requested model by calling `GET /v1/models`."""
        try:
            available_models: List[str] = []

            # Backward-compatible path used by unit tests that mock SDK style.
            if hasattr(client, "models") and hasattr(client.models, "list") and callable(client.models.list):
                models_response = client.models.list()
                available_models = self._extract_model_ids_from_items(getattr(models_response, "data", []))
            else:
                response = client.get("/v1/models")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    raw_data = payload.get("data", [])
                    data: List[Any] = raw_data if isinstance(raw_data, list) else []
                elif isinstance(payload, list):
                    data = payload
                else:
                    data = []
                available_models = self._extract_model_ids_from_items(data)

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Anthropic",
                    available_models=available_models,
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)

    async def _validate_model_async(self, async_client: httpx.AsyncClient, model_id: str) -> None:
        """Async variant of model validation with `GET /v1/models`."""
        try:
            available_models: List[str] = []

            # Backward-compatible path used by unit tests that mock SDK style.
            if hasattr(async_client, "models") and hasattr(async_client.models, "list") and callable(async_client.models.list):
                models_response = await async_client.models.list()
                available_models = self._extract_model_ids_from_items(getattr(models_response, "data", []))
            else:
                response = await async_client.get("/v1/models")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    raw_data = payload.get("data", [])
                    data: List[Any] = raw_data if isinstance(raw_data, list) else []
                elif isinstance(payload, list):
                    data = payload
                else:
                    data = []
                available_models = self._extract_model_ids_from_items(data)

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Anthropic",
                    available_models=available_models,
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)

    def _parse_sse_line_payload(self, event_name: Optional[str], data_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse buffered SSE payload for one event."""
        if not data_lines:
            return None

        data_str = "\n".join(data_lines).strip()
        if not data_str:
            return None
        if data_str == "[DONE]":
            return {"type": "[DONE]"}

        try:
            payload = json_loads(data_str)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if "type" not in payload and event_name:
            payload["type"] = event_name
        return payload

    def _iter_sse_events(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Yield parsed SSE events from a sync HTTPX streaming response."""
        event_name: Optional[str] = None
        data_lines: List[str] = []

        for line in response.iter_lines():
            if line is None:
                continue
            stripped = line.strip()
            if stripped == "":
                payload = self._parse_sse_line_payload(event_name, data_lines)
                if payload is not None:
                    yield payload
                event_name = None
                data_lines = []
                continue
            if stripped.startswith(":"):
                continue
            if stripped.startswith("event:"):
                event_name = stripped[6:].strip()
                continue
            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].strip())

        payload = self._parse_sse_line_payload(event_name, data_lines)
        if payload is not None:
            yield payload

    async def _aiter_sse_events(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield parsed SSE events from an async HTTPX streaming response."""
        event_name: Optional[str] = None
        data_lines: List[str] = []

        async for line in response.aiter_lines():
            if line is None:
                continue
            stripped = line.strip()
            if stripped == "":
                payload = self._parse_sse_line_payload(event_name, data_lines)
                if payload is not None:
                    yield payload
                event_name = None
                data_lines = []
                continue
            if stripped.startswith(":"):
                continue
            if stripped.startswith("event:"):
                event_name = stripped[6:].strip()
                continue
            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].strip())

        payload = self._parse_sse_line_payload(event_name, data_lines)
        if payload is not None:
            yield payload


class AsyncAnthropicProvider(AsyncBaseProvider, _AnthropicProviderMixin):
    """Asynchronous Anthropic provider implemented with HTTPX."""

    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> None:
        (
            _api_key,
            client_kwargs,
            self.model_id,
            self.default_params,
            self.max_retries,
        ) = self._init_common(model_id, api_key, **kwargs)

        try:
            self.async_client = httpx.AsyncClient(**client_kwargs)
        except Exception as e:
            raise ConnectionError(provider="Anthropic") from e

    async def validate_model(self) -> None:
        """Validate configured model is available via Anthropic API."""
        await self._validate_model_async(async_client=self.async_client, model_id=self.model_id)

    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)

    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        return self._parse_tool_call_sync(tool_call)

    async def _format_tool_call(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._format_tool_call_sync(tools)

    async def _prepare_chat_history(
        self,
        chat_history: List[Dict[str, Any]],
        *,
        enable_cache: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self._prepare_chat_history_sync(chat_history, enable_cache=enable_cache)

    async def async_generate(
        self,
        chat_history: List[
            Union[
                Dict[str, Any],
                ToolCallRequest,
                ToolCallResult,
                ToolCallError,
                AssistantMessage,
                UserMessage,
                SystemMessage,
            ]
        ],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Asynchronously generate an Anthropic message completion."""
        params = self._prepare_params(**kwargs)

        if tools is not None:
            params["tools"] = await self._format_tool_call(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        system_message: List[Dict[str, Any]] = []
        formatted_chat_history: List[Dict[str, Any]] = []
        if chat_history:
            system_message, formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": formatted_chat_history,
            **params,
        }
        if system_message:
            payload["system"] = system_message

        async def _generate() -> Union[LLMResponse, ToolCallList]:
            response = await self.async_client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise LLMError(message="Anthropic API returned an unexpected response payload.")

            (
                text_content,
                thinking_content,
                tool_calls,
                stop_reason,
                model_used,
                usage_data,
            ) = self._extract_message_output(data, effective_model)

            if tool_calls:
                return ToolCallList(
                    model_id=model_used,
                    tool_calls=await self._parse_tool_call(tool_calls),
                    usage=usage_data,
                )

            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text_content,
                thinking=thinking_content,
                usage=usage_data,
                stop_reason=stop_reason,
            )

        return await self._execute_with_retry(operation=_generate)

    async def async_stream(
        self,
        chat_history: List[
            Union[
                Dict[str, Any],
                ToolCallRequest,
                ToolCallResult,
                ToolCallError,
                AssistantMessage,
                UserMessage,
                SystemMessage,
            ]
        ],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[str, LLMResponse, ToolCallList], None]:
        """Asynchronously stream Anthropic message deltas and final response."""
        params = self._prepare_params(**kwargs)

        if tools is not None:
            params["tools"] = await self._format_tool_call(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        system_message: List[Dict[str, Any]] = []
        formatted_chat_history: List[Dict[str, Any]] = []
        if chat_history:
            system_message, formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": formatted_chat_history,
            "stream": True,
            **params,
        }
        if system_message:
            payload["system"] = system_message

        async def _open_stream():
            stream_ctx = self.async_client.stream("POST", "/v1/messages", json=payload)
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
        tool_calls: List[ToolCall] = []
        current_tool_call: Optional[ToolCall] = None
        current_tool_input_parts: List[str] = []
        current_block_type: Optional[str] = None

        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: Optional[str] = None
        final_model = effective_model

        try:
            stream_ctx, response = await self._execute_with_retry(operation=_open_stream)
            try:
                async for event in self._aiter_sse_events(response):
                    event_type = event.get("type")
                    if event_type == "[DONE]":
                        break

                    if event_type == "error":
                        err = event.get("error", {})
                        if isinstance(err, dict):
                            raise LLMError(message=str(err.get("message", "Anthropic streaming error.")))
                        raise LLMError(message="Anthropic streaming error.")

                    if event_type == "message_start":
                        message = event.get("message", {})
                        if isinstance(message, dict):
                            final_model = str(message.get("model", final_model))
                            usage = message.get("usage", {})
                            if isinstance(usage, dict):
                                prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                        continue

                    if event_type == "content_block_start":
                        content_block = event.get("content_block", {})
                        if not isinstance(content_block, dict):
                            continue
                        current_block_type = content_block.get("type")
                        if current_block_type == "tool_use":
                            initial_args = content_block.get("input", {})
                            if isinstance(initial_args, str):
                                try:
                                    initial_args = json_loads(initial_args)
                                except Exception:
                                    initial_args = {}
                            current_tool_call = ToolCall(
                                id=str(content_block.get("id", "")),
                                type="function_call",
                                function_name=str(content_block.get("name", "")),
                                arguments=initial_args if isinstance(initial_args, dict) else {},
                            )
                            current_tool_input_parts = []
                        continue

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if not isinstance(delta, dict):
                            continue

                        delta_type = delta.get("type")
                        if delta_type == "text_delta" or (current_block_type == "text" and "text" in delta):
                            delta_text = delta.get("text", "")
                            if delta_text:
                                completion_parts.append(delta_text)
                                yield LLMResponse(
                                    model_id=final_model,
                                    type="llm_delta",
                                    text=delta_text,
                                    thinking=None,
                                    usage=None,
                                    stop_reason=None,
                                )
                            continue

                        if delta_type == "thinking_delta" or (current_block_type == "thinking" and "thinking" in delta):
                            delta_thinking = delta.get("thinking", "")
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
                            continue

                        if delta_type == "input_json_delta" and current_tool_call:
                            partial_json = delta.get("partial_json", "")
                            if partial_json:
                                current_tool_input_parts.append(partial_json)
                            continue

                        continue

                    if event_type == "content_block_stop":
                        if current_tool_call:
                            if current_tool_input_parts:
                                try:
                                    parsed = json_loads("".join(current_tool_input_parts))
                                    if isinstance(parsed, dict):
                                        current_tool_call.arguments = parsed
                                    else:
                                        current_tool_call.arguments = {}
                                except Exception:
                                    current_tool_call.arguments = current_tool_call.arguments or {}
                            tool_calls.append(current_tool_call)
                            current_tool_call = None
                            current_tool_input_parts = []
                        current_block_type = None
                        continue

                    if event_type == "message_delta":
                        usage = event.get("usage", {})
                        if isinstance(usage, dict):
                            completion_tokens = int(usage.get("output_tokens", completion_tokens) or completion_tokens)
                        delta = event.get("delta", {})
                        if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                            stop_reason = delta.get("stop_reason")
                        continue
            finally:
                await stream_ctx.__aexit__(None, None, None)

            if current_tool_call:
                if current_tool_input_parts:
                    try:
                        parsed = json_loads("".join(current_tool_input_parts))
                        if isinstance(parsed, dict):
                            current_tool_call.arguments = parsed
                    except Exception:
                        pass
                tool_calls.append(current_tool_call)

            usage_obj = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            if tool_calls:
                yield ToolCallList(model_id=final_model, tool_calls=tool_calls, usage=usage_obj)
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
            raise self._handle_error(e)


class AnthropicProvider(BaseProvider, _AnthropicProviderMixin):
    """Synchronous Anthropic provider implemented with HTTPX."""

    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> None:
        (
            _api_key,
            client_kwargs,
            self.model_id,
            self.default_params,
            self.max_retries,
        ) = self._init_common(model_id, api_key, **kwargs)

        try:
            self.client = httpx.Client(**client_kwargs)
        except Exception as e:
            raise ConnectionError(provider="Anthropic") from e

        try:
            self._validate_model_sync(client=self.client, model_id=self.model_id)
        except BaseException:
            # Construction failed, so no caller can close the owned client.
            self.client.close()
            raise

    def validate_model(self) -> None:
        """Validate configured model is available via Anthropic API."""
        self._validate_model_sync(client=self.client, model_id=self.model_id)

    def generate(
        self,
        chat_history: List[
            Union[
                Dict[str, Any],
                ToolCallRequest,
                ToolCallResult,
                ToolCallError,
                AssistantMessage,
                UserMessage,
                SystemMessage,
            ]
        ],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Synchronously generate an Anthropic message completion."""
        params = self._prepare_params(**kwargs)

        if tools is not None:
            params["tools"] = self._format_tool_call_sync(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        system_message: List[Dict[str, Any]] = []
        formatted_chat_history: List[Dict[str, Any]] = []
        if chat_history:
            system_message, formatted_chat_history = self._prepare_chat_history_sync(chat_history=chat_history)

        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": formatted_chat_history,
            **params,
        }
        if system_message:
            payload["system"] = system_message

        def _generate() -> Union[LLMResponse, ToolCallList]:
            response = self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise LLMError(message="Anthropic API returned an unexpected response payload.")

            (
                text_content,
                thinking_content,
                tool_calls,
                stop_reason,
                model_used,
                usage_data,
            ) = self._extract_message_output(data, effective_model)

            if tool_calls:
                return ToolCallList(
                    model_id=model_used,
                    tool_calls=self._parse_tool_call_sync(tool_calls),
                    usage=usage_data,
                )

            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text_content,
                thinking=thinking_content,
                usage=usage_data,
                stop_reason=stop_reason,
            )

        return self._execute_with_retry(operation=_generate)

    def stream(
        self,
        chat_history: List[
            Union[
                Dict[str, Any],
                ToolCallRequest,
                ToolCallResult,
                ToolCallError,
                AssistantMessage,
                UserMessage,
                SystemMessage,
            ]
        ],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Union[str, LLMResponse, ToolCallList], None, None]:
        """Synchronously stream Anthropic message deltas and final response."""
        params = self._prepare_params(**kwargs)

        if tools is not None:
            params["tools"] = self._format_tool_call_sync(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        system_message: List[Dict[str, Any]] = []
        formatted_chat_history: List[Dict[str, Any]] = []
        if chat_history:
            system_message, formatted_chat_history = self._prepare_chat_history_sync(chat_history=chat_history)

        effective_model = model or self.model_id
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": formatted_chat_history,
            "stream": True,
            **params,
        }
        if system_message:
            payload["system"] = system_message

        def _open_stream():
            stream_ctx = self.client.stream("POST", "/v1/messages", json=payload)
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
        tool_calls: List[ToolCall] = []
        current_tool_call: Optional[ToolCall] = None
        current_tool_input_parts: List[str] = []
        current_block_type: Optional[str] = None

        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: Optional[str] = None
        final_model = effective_model

        try:
            stream_ctx, response = self._execute_with_retry(operation=_open_stream)
            try:
                for event in self._iter_sse_events(response):
                    event_type = event.get("type")
                    if event_type == "[DONE]":
                        break

                    if event_type == "error":
                        err = event.get("error", {})
                        if isinstance(err, dict):
                            raise LLMError(message=str(err.get("message", "Anthropic streaming error.")))
                        raise LLMError(message="Anthropic streaming error.")

                    if event_type == "message_start":
                        message = event.get("message", {})
                        if isinstance(message, dict):
                            final_model = str(message.get("model", final_model))
                            usage = message.get("usage", {})
                            if isinstance(usage, dict):
                                prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                        continue

                    if event_type == "content_block_start":
                        content_block = event.get("content_block", {})
                        if not isinstance(content_block, dict):
                            continue
                        current_block_type = content_block.get("type")
                        if current_block_type == "tool_use":
                            initial_args = content_block.get("input", {})
                            if isinstance(initial_args, str):
                                try:
                                    initial_args = json_loads(initial_args)
                                except Exception:
                                    initial_args = {}
                            current_tool_call = ToolCall(
                                id=str(content_block.get("id", "")),
                                type="function_call",
                                function_name=str(content_block.get("name", "")),
                                arguments=initial_args if isinstance(initial_args, dict) else {},
                            )
                            current_tool_input_parts = []
                        continue

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if not isinstance(delta, dict):
                            continue

                        delta_type = delta.get("type")
                        if delta_type == "text_delta" or (current_block_type == "text" and "text" in delta):
                            delta_text = delta.get("text", "")
                            if delta_text:
                                completion_parts.append(delta_text)
                                yield LLMResponse(
                                    model_id=final_model,
                                    type="llm_delta",
                                    text=delta_text,
                                    thinking=None,
                                    usage=None,
                                    stop_reason=None,
                                )
                            continue

                        if delta_type == "thinking_delta" or (current_block_type == "thinking" and "thinking" in delta):
                            delta_thinking = delta.get("thinking", "")
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
                            continue

                        if delta_type == "input_json_delta" and current_tool_call:
                            partial_json = delta.get("partial_json", "")
                            if partial_json:
                                current_tool_input_parts.append(partial_json)
                            continue

                        continue

                    if event_type == "content_block_stop":
                        if current_tool_call:
                            if current_tool_input_parts:
                                try:
                                    parsed = json_loads("".join(current_tool_input_parts))
                                    if isinstance(parsed, dict):
                                        current_tool_call.arguments = parsed
                                    else:
                                        current_tool_call.arguments = {}
                                except Exception:
                                    current_tool_call.arguments = current_tool_call.arguments or {}
                            tool_calls.append(current_tool_call)
                            current_tool_call = None
                            current_tool_input_parts = []
                        current_block_type = None
                        continue

                    if event_type == "message_delta":
                        usage = event.get("usage", {})
                        if isinstance(usage, dict):
                            completion_tokens = int(usage.get("output_tokens", completion_tokens) or completion_tokens)
                        delta = event.get("delta", {})
                        if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                            stop_reason = delta.get("stop_reason")
                        continue
            finally:
                stream_ctx.__exit__(None, None, None)

            if current_tool_call:
                if current_tool_input_parts:
                    try:
                        parsed = json_loads("".join(current_tool_input_parts))
                        if isinstance(parsed, dict):
                            current_tool_call.arguments = parsed
                    except Exception:
                        pass
                tool_calls.append(current_tool_call)

            usage_obj = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            if tool_calls:
                yield ToolCallList(model_id=final_model, tool_calls=tool_calls, usage=usage_obj)
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
            raise self._handle_error(e)

