"""OpenAI provider implementation using HTTPX + Responses API."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import httpx

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

_EMPTY_JSON: str = "{}"


class _OpenAIProviderMixin:
    """Shared functionality between sync and async OpenAI providers."""

    def _init_common(self, model_id: str, api_key: Optional[str], base_url: Optional[str], **kwargs) -> tuple:
        """Common initialization logic shared between sync and async providers."""
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(message="OpenAI API key not provided", missing_vars=["OPENAI_API_KEY"])

        if not model_id:
            raise InvalidRequestError(message="model_id is required for OpenAI provider")

        max_retries = kwargs.get("max_retries", 3)
        timeout = kwargs.get("timeout", 30.0)
        effective_base_url = (base_url or kwargs.get("base_url") or "https://api.openai.com/v1").rstrip("/")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        default_params: Dict[str, Any] = {}
        valid_query_keys = {
            "background",
            "context_management",
            "conversation",
            "include",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
            "reasoning",
            "safety_identifier",
            "service_tier",
            "store",
            "stream_options",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        }
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value

        return effective_base_url, timeout, model_id, default_params, max_retries, headers

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters by merging defaults with overrides."""
        params = self.default_params.copy()
        params.update(kwargs)
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def _model_disallowed_params(model_id: str) -> set[str]:
        """Return request keys that should not be sent for a model family."""
        normalized = str(model_id or "").lower()
        if normalized.startswith("gpt-5"):
            return {"temperature"}
        return set()

    def _sanitize_params_for_model(self, model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Drop params that are known to be rejected by the target model."""
        disallowed = self._model_disallowed_params(model_id)
        if not disallowed:
            return params
        return {k: v for k, v in params.items() if k not in disallowed}

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
        response = error.response
        status_code = response.status_code
        message = str(error)
        code = None
        param = None

        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    message = err.get("message", message)
                    code = err.get("code")
                    param = err.get("param")
                elif isinstance(err, str) and err.strip():
                    message = err
                elif payload.get("message"):
                    message = str(payload["message"])
        except Exception:
            pass

        # Fallback to raw body text if provider payload couldn't be parsed.
        if message == str(error) or message.lower().startswith(("client error", "server error")):
            raw_body = self._safe_response_text(response)
            if raw_body:
                message = raw_body

        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if request_id and f"request_id={request_id}" not in message:
            message = f"{message} (request_id={request_id})"

        return status_code, message, code, param

    def _handle_error(self, error: Exception) -> Exception:
        """Map HTTPX/OpenAI Responses errors to standard provider errors."""
        if isinstance(error, LLMError):
            return error

        if isinstance(error, httpx.HTTPStatusError):
            status_code, message, code, param = self._extract_error_details(error)
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
                    numbers = [int(n) for n in re.findall(r"\d+", message)]
                    tokens = numbers[0] if numbers else None
                    max_tokens = numbers[1] if len(numbers) > 1 else None
                    return ContextLengthExceededError(
                        model_id=self.model_id, tokens=tokens, max_tokens=max_tokens
                    )
                if "content_policy_violation" in code_l or "image_content_policy_violation" in code_l:
                    return ContentFilterError(provider="OpenAI", filter_reason=message)
                return InvalidRequestError(message=message, param=param)
            if status_code == 403:
                return InvalidRequestError(message=f"OpenAI permission denied: {message}", param=param)
            if status_code == 422:
                return ContentFilterError(provider="OpenAI", filter_reason=message)
            if 500 <= status_code < 600:
                return ServerError(provider="OpenAI", status_code=status_code)
            return LLMError(message=f"OpenAI API Error {status_code}: {message}")

        if isinstance(error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return ConnectionError(provider="OpenAI", message="Request timed out")
        if isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).lower():
            return ConnectionError(provider="OpenAI", message="Request timed out")
        if isinstance(error, (httpx.NetworkError, httpx.TransportError, httpx.ConnectError)):
            return ConnectionError(provider="OpenAI", message=f"Connection error: {error}")
        if isinstance(error, httpx.HTTPError):
            return ConnectionError(provider="OpenAI", message=f"HTTP error: {error}")

        return LLMError(message=f"An unexpected OpenAI error occurred: {error}")

    def _convert_image_content_responses(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert image content blocks to OpenAI Responses API format."""
        converted: List[Dict[str, Any]] = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image":
                if "url" in block:
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": block["url"],
                            "detail": block.get("detail", "auto"),
                        }
                    )
                elif "base64" in block:
                    media_type = block.get("media_type", "image/jpeg")
                    data_url = f"data:{media_type};base64,{block['base64']}"
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": block.get("detail", "auto"),
                        }
                    )
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        "detail": image_url.get("detail", "auto")
                        if isinstance(image_url, dict)
                        else "auto",
                    }
                )
            elif block_type in {"input_text", "input_image", "input_file"}:
                converted.append(block)
            elif block_type == "file":
                # OpenAI chat-style file block => Responses API input_file
                file_obj = block.get("file", {})
                converted.append(
                    {
                        "type": "input_file",
                        "file_data": file_obj.get("file_data"),
                        "file_id": file_obj.get("file_id"),
                        "filename": file_obj.get("filename"),
                    }
                )
            else:
                if "text" in block:
                    converted.append({"type": "input_text", "text": block["text"]})
                else:
                    converted.append(block)

        return converted

    @staticmethod
    def _make_strict_json_schema(schema: Any) -> Any:  # noqa: ANN401
        """Return a strict-compliant JSON schema copy for OpenAI tool parameters."""
        if isinstance(schema, list):
            return [_OpenAIProviderMixin._make_strict_json_schema(x) for x in schema]
        if not isinstance(schema, dict):
            return schema

        out: Dict[str, Any] = {k: _OpenAIProviderMixin._make_strict_json_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            props = out.get("properties")
            if isinstance(props, dict) and props:
                out["additionalProperties"] = False
        return out

    def _prepare_input_responses(
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
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Prepare request input for the OpenAI Responses API."""
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        def _is_responses_function_id(value: Any) -> bool:
            return isinstance(value, str) and value.startswith("fc_")

        for raw_msg in chat_history:
            if isinstance(raw_msg, ToolCallRequest):
                call_id = str(raw_msg.id)
                item: Dict[str, Any] = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": raw_msg.name,
                    "arguments": json_dumps(raw_msg.arguments) if raw_msg.arguments else _EMPTY_JSON,
                }
                # Responses API item IDs use the "fc_" prefix. Internal tool-call IDs
                # are typically "call_*" and should not be sent as item IDs.
                if _is_responses_function_id(raw_msg.id):
                    item["id"] = str(raw_msg.id)
                input_items.append(
                    item
                )
                continue

            if isinstance(raw_msg, (ToolCallResult, ToolCallError)):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(raw_msg.tool_call_id),
                        "output": str(raw_msg.error if isinstance(raw_msg, ToolCallError) else raw_msg.result),
                    }
                )
                continue

            if isinstance(raw_msg, SystemMessage):
                instructions_parts.append(raw_msg.content)
                continue

            if isinstance(raw_msg, (AssistantMessage, UserMessage)):
                content = raw_msg.content
                role = raw_msg.role
                api_role = "assistant" if role == "assistant" else "user"

                if isinstance(content, list):
                    content = self._convert_image_content_responses(content)
                input_items.append({"type": "message", "role": api_role, "content": content})
                continue

            role = raw_msg.get("role")
            if role in {"system", "developer"}:
                content = raw_msg.get("content", "")
                instructions_parts.append(content if isinstance(content, str) else str(content))

            elif role == "tool_call":
                raw_id = raw_msg.get("id")
                raw_call_id = raw_msg.get("call_id")
                call_id = raw_call_id if raw_call_id is not None else raw_id
                item = {
                    "type": "function_call",
                    "call_id": str(call_id) if call_id is not None else "",
                    "name": raw_msg.get("name"),
                    "arguments": json_dumps(raw_msg.get("arguments", {}))
                    if raw_msg.get("arguments")
                    else _EMPTY_JSON,
                }
                if _is_responses_function_id(raw_id):
                    item["id"] = str(raw_id)
                input_items.append(
                    item
                )

            elif role in {"tool_result", "tool_error", "tool"}:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(raw_msg.get("tool_call_id")),
                        "output": str(raw_msg.get("error", "") if role == "tool_error" else raw_msg.get("result", raw_msg.get("content", ""))),
                    }
                )

            elif role in {"assistant", "user"}:
                content = raw_msg.get("content")
                if isinstance(content, list):
                    content = self._convert_image_content_responses(content)
                input_items.append({"type": "message", "role": role, "content": content})

        instructions = "\n".join(instructions_parts) if instructions_parts else None
        return instructions, input_items

    def _format_tools_responses(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI Responses API."""
        formatted_tools: List[Dict[str, Any]] = []

        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                func_def = tool.get("function", tool)
                parameters = func_def.get("parameters", {}) or {}
                strict = func_def.get("strict")

                if isinstance(parameters, dict) and parameters.get("type") == "object":
                    props = parameters.get("properties")
                    if isinstance(props, dict) and props:
                        parameters = self._make_strict_json_schema(parameters)
                    elif strict is True:
                        # Mapping-style object schemas are incompatible with strict mode.
                        strict = False

                tool_payload: Dict[str, Any] = {
                    "type": "function",
                    "name": func_def.get("name"),
                    "description": func_def.get("description", ""),
                }
                if parameters is not None:
                    tool_payload["parameters"] = parameters
                if strict is not None:
                    tool_payload["strict"] = strict
                formatted_tools.append(tool_payload)
            else:
                formatted_tools.append(tool)

        return formatted_tools

    def _parse_tool_call_sync(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        """Synchronous version of tool call parsing."""
        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            args = tc.get("arguments", {})
            try:
                args = json_loads(args) if isinstance(args, str) else args
            except Exception:
                args = {}

            processed_tool_calls.append(
                ToolCall(
                    id=tc.get("call_id", tc.get("id", "")),
                    type="function_call",
                    function_name=tc.get("function_name", tc.get("name", "")),
                    arguments=args or {},
                )
            )

        return processed_tool_calls

    def _extract_response_content(
        self, response: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Extract text/thinking/tool calls from a Responses API JSON payload."""
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = response.get("status")

        output = response.get("output") or []
        if isinstance(output, list):
            for output_item in output:
                if not isinstance(output_item, dict):
                    continue
                item_type = output_item.get("type")

                if item_type == "message":
                    content = output_item.get("content") or []
                    if isinstance(content, list):
                        for content_block in content:
                            if not isinstance(content_block, dict):
                                continue
                            content_type = content_block.get("type")
                            if content_type == "output_text":
                                text_value = content_block.get("text")
                                if text_value:
                                    text_parts.append(text_value)
                            elif content_type == "refusal":
                                refusal = content_block.get("refusal")
                                if refusal:
                                    text_parts.append(refusal)

                elif item_type == "reasoning":
                    summary = output_item.get("summary") or []
                    if isinstance(summary, list):
                        for summary_block in summary:
                            if isinstance(summary_block, dict) and summary_block.get("text"):
                                thinking_parts.append(summary_block["text"])
                    content = output_item.get("content") or []
                    if isinstance(content, list):
                        for reasoning_block in content:
                            if isinstance(reasoning_block, dict) and reasoning_block.get("text"):
                                thinking_parts.append(reasoning_block["text"])

                elif item_type == "function_call":
                    call_id = output_item.get("call_id", output_item.get("id", ""))
                    item_id = output_item.get("id", "")
                    name = output_item.get("name", "")
                    arguments = output_item.get("arguments", _EMPTY_JSON)
                    tool_calls.append(
                        {
                            "id": item_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": arguments,
                        }
                    )

        text_content = "".join(text_parts) if text_parts else response.get("output_text")
        thinking_content = "".join(thinking_parts) if thinking_parts else None
        return text_content, thinking_content, tool_calls, stop_reason

    def _extract_usage(self, response: Dict[str, Any]) -> LLMUsage:
        """Extract usage information from OpenAI Responses API payload."""
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _extract_model_ids_from_items(items: Any) -> List[str]:
        """Extract model IDs from a JSON model listing."""
        if not isinstance(items, list):
            return []
        return [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]

    def _validate_model_sync(self, client: httpx.Client, model_id: str) -> None:
        """Validate that the requested model is available via `/models`."""
        try:
            response = client.get("/models")
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict):
                data = payload.get("data", [])
            elif isinstance(payload, list):
                data = payload
            else:
                data = []

            available_models = self._extract_model_ids_from_items(data)

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id, provider="OpenAI", available_models=available_models
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)

    async def _validate_model_async(self, async_client: httpx.AsyncClient, model_id: str) -> None:
        """Asynchronously validate that requested model is available via `/models`."""
        try:
            response = await async_client.get("/models")
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict):
                data = payload.get("data", [])
            elif isinstance(payload, list):
                data = payload
            else:
                data = []

            available_models = self._extract_model_ids_from_items(data)

            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id, provider="OpenAI", available_models=available_models
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise self._handle_error(e)

    def _parse_sse_line_payload(self, event_name: Optional[str], data_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse buffered SSE data payload into a dict event."""
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


class AsyncOpenAIProvider(AsyncBaseProvider, _OpenAIProviderMixin):
    """Asynchronous provider implementation for OpenAI Responses API via HTTPX."""

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
            headers,
        ) = self._init_common(model_id, api_key, base_url, **kwargs)

        try:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="OpenAI") from e

    async def validate_model(self) -> None:
        """Validate that configured model is available via API."""
        await self._validate_model_async(async_client=self.async_client, model_id=self.model_id)

    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)

    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        return self._parse_tool_call_sync(tool_call)

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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Asynchronously generate a completion for provided chat history."""
        params = self._prepare_params(**kwargs)
        instructions, input_items = self._prepare_input_responses(chat_history)

        if instructions:
            params["instructions"] = instructions
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        params = self._sanitize_params_for_model(effective_model, params)
        payload: Dict[str, Any] = {"model": effective_model, **params}
        if input_items:
            payload["input"] = input_items

        async def _generate() -> Union[LLMResponse, ToolCallList]:
            response = await self.async_client.post("/responses", json=payload)
            response.raise_for_status()
            data = response.json()

            text_content, thinking_content, tool_calls, stop_reason = self._extract_response_content(data)
            usage_obj = self._extract_usage(data)
            model_used = data.get("model", effective_model)

            if tool_calls:
                tool_calls_parsed = await self._parse_tool_call(tool_calls)
                return ToolCallList(model_id=model_used, tool_calls=tool_calls_parsed, usage=usage_obj)

            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text_content,
                thinking=thinking_content,
                usage=usage_obj,
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """Asynchronously stream a completion for provided chat history."""
        params = self._prepare_params(**kwargs)
        instructions, input_items = self._prepare_input_responses(chat_history)

        if instructions:
            params["instructions"] = instructions
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        params = self._sanitize_params_for_model(effective_model, params)
        payload: Dict[str, Any] = {"model": effective_model, "stream": True, **params}
        if input_items:
            payload["input"] = input_items

        async def _open_stream():
            stream_ctx = self.async_client.stream("POST", "/responses", json=payload)
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
        final_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stop_reason: Optional[str] = None
        final_model = effective_model
        completed_response: Optional[Dict[str, Any]] = None

        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        tool_call_order: List[str] = []
        tool_calls_no_id: List[Dict[str, Any]] = []
        current_tool_call: Optional[Dict[str, Any]] = None
        current_tool_args_parts: List[str] = []

        def _track_tool_call(call: Dict[str, Any]) -> None:
            cid = str(call.get("call_id") or call.get("id") or "")
            if cid:
                if cid not in tool_calls_by_id:
                    tool_call_order.append(cid)
                tool_calls_by_id[cid] = call
            else:
                tool_calls_no_id.append(call)

        try:
            stream_ctx, response = await self._execute_with_retry(operation=_open_stream)
            try:
                async for event in self._aiter_sse_events(response):
                    event_type = event.get("type")

                    if event_type == "[DONE]":
                        break

                    if event_type == "response.output_text.delta":
                        delta_text = event.get("delta", "")
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

                    if event_type in {"response.reasoning.delta", "response.reasoning_summary_text.delta"}:
                        delta_thinking = event.get("delta", "")
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

                    if event_type == "response.function_call_arguments.start":
                        call_id = event.get("call_id", event.get("id", ""))
                        name = event.get("name", "")
                        current_tool_call = {
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": "",
                        }
                        current_tool_args_parts = []
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        if current_tool_call:
                            delta_args = event.get("delta", "")
                            if delta_args:
                                current_tool_args_parts.append(delta_args)
                        continue

                    if event_type == "response.function_call_arguments.done":
                        if current_tool_call:
                            if current_tool_args_parts:
                                current_tool_call["arguments"] = "".join(current_tool_args_parts)
                            elif event.get("arguments"):
                                current_tool_call["arguments"] = event.get("arguments")
                            _track_tool_call(current_tool_call)
                            current_tool_call = None
                            current_tool_args_parts = []
                        continue

                    if event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id", item.get("id", ""))
                            name = item.get("name", "")
                            initial_args = item.get("arguments", "") or ""
                            current_tool_call = {
                                "id": call_id,
                                "call_id": call_id,
                                "name": name,
                                "function_name": name,
                                "arguments": initial_args,
                            }
                            current_tool_args_parts = [initial_args] if initial_args else []
                        continue

                    if event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id", item.get("id", ""))
                            name = item.get("name", "")
                            arguments = item.get("arguments", _EMPTY_JSON)
                            _track_tool_call(
                                {
                                    "id": call_id,
                                    "call_id": call_id,
                                    "name": name,
                                    "function_name": name,
                                    "arguments": arguments,
                                }
                            )
                            current_tool_call = None
                            current_tool_args_parts = []
                        continue

                    if event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            completed_response = response_payload
                            final_model = response_payload.get("model", final_model)
                            stop_reason = response_payload.get("status", stop_reason)
                            usage_obj = self._extract_usage(response_payload)
                            final_usage_data = {
                                "prompt_tokens": usage_obj.prompt_tokens,
                                "completion_tokens": usage_obj.completion_tokens,
                                "total_tokens": usage_obj.total_tokens,
                            }
                        continue
            finally:
                await stream_ctx.__aexit__(None, None, None)

            # Prefer final completed payload if present; otherwise use tracked deltas.
            if completed_response is not None:
                text_content, thinking_content, completed_tool_calls, stop_reason = self._extract_response_content(
                    completed_response
                )
                usage_obj = self._extract_usage(completed_response)
                final_model = completed_response.get("model", final_model)
                tool_calls: List[Dict[str, Any]] = completed_tool_calls
                if not tool_calls:
                    tool_calls = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id

                if tool_calls:
                    tool_calls_obj = await self._parse_tool_call(tool_calls)
                    yield ToolCallList(model_id=final_model, tool_calls=tool_calls_obj, usage=usage_obj)
                else:
                    yield LLMResponse(
                        model_id=final_model,
                        type="llm_response",
                        text=text_content if text_content is not None else ("".join(completion_parts) or None),
                        thinking=thinking_content
                        if thinking_content is not None
                        else ("".join(thinking_parts) or None),
                        usage=usage_obj,
                        stop_reason=stop_reason,
                    )
            else:
                usage_obj = LLMUsage(**final_usage_data)
                tool_calls = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id
                if tool_calls:
                    tool_calls_obj = await self._parse_tool_call(tool_calls)
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
            raise self._handle_error(e)


class OpenAIProvider(BaseProvider, _OpenAIProviderMixin):
    """Synchronous provider implementation for OpenAI Responses API via HTTPX."""

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
            headers,
        ) = self._init_common(model_id, api_key, base_url, **kwargs)

        try:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                http2=True,
            )
        except Exception as e:
            raise ConnectionError(provider="OpenAI") from e

        try:
            self._validate_model_sync(client=self.client, model_id=self.model_id)
        except BaseException:
            # Construction failed, so no caller can close the owned client.
            self.client.close()
            raise

    def validate_model(self) -> None:
        """Validate that configured model is available via API."""
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """Synchronously generate a completion for provided chat history."""
        params = self._prepare_params(**kwargs)
        instructions, input_items = self._prepare_input_responses(chat_history)

        if instructions:
            params["instructions"] = instructions
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        params = self._sanitize_params_for_model(effective_model, params)
        payload: Dict[str, Any] = {"model": effective_model, **params}
        if input_items:
            payload["input"] = input_items

        def _generate() -> Union[LLMResponse, ToolCallList]:
            response = self.client.post("/responses", json=payload)
            response.raise_for_status()
            data = response.json()

            text_content, thinking_content, tool_calls, stop_reason = self._extract_response_content(data)
            usage_obj = self._extract_usage(data)
            model_used = data.get("model", effective_model)

            if tool_calls:
                tool_calls_parsed = self._parse_tool_call_sync(tool_calls)
                return ToolCallList(model_id=model_used, tool_calls=tool_calls_parsed, usage=usage_obj)

            return LLMResponse(
                model_id=model_used,
                type="llm_response",
                text=text_content,
                thinking=thinking_content,
                usage=usage_obj,
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        """Synchronously stream a completion for provided chat history."""
        params = self._prepare_params(**kwargs)
        instructions, input_items = self._prepare_input_responses(chat_history)

        if instructions:
            params["instructions"] = instructions
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        params = self._sanitize_params_for_model(effective_model, params)
        payload: Dict[str, Any] = {"model": effective_model, "stream": True, **params}
        if input_items:
            payload["input"] = input_items

        def _open_stream():
            stream_ctx = self.client.stream("POST", "/responses", json=payload)
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
        final_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stop_reason: Optional[str] = None
        final_model = effective_model
        completed_response: Optional[Dict[str, Any]] = None

        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        tool_call_order: List[str] = []
        tool_calls_no_id: List[Dict[str, Any]] = []
        current_tool_call: Optional[Dict[str, Any]] = None
        current_tool_args_parts: List[str] = []

        def _track_tool_call(call: Dict[str, Any]) -> None:
            cid = str(call.get("call_id") or call.get("id") or "")
            if cid:
                if cid not in tool_calls_by_id:
                    tool_call_order.append(cid)
                tool_calls_by_id[cid] = call
            else:
                tool_calls_no_id.append(call)

        try:
            stream_ctx, response = self._execute_with_retry(operation=_open_stream)
            try:
                for event in self._iter_sse_events(response):
                    event_type = event.get("type")

                    if event_type == "[DONE]":
                        break

                    if event_type == "response.output_text.delta":
                        delta_text = event.get("delta", "")
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

                    if event_type in {"response.reasoning.delta", "response.reasoning_summary_text.delta"}:
                        delta_thinking = event.get("delta", "")
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

                    if event_type == "response.function_call_arguments.start":
                        call_id = event.get("call_id", event.get("id", ""))
                        name = event.get("name", "")
                        current_tool_call = {
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": "",
                        }
                        current_tool_args_parts = []
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        if current_tool_call:
                            delta_args = event.get("delta", "")
                            if delta_args:
                                current_tool_args_parts.append(delta_args)
                        continue

                    if event_type == "response.function_call_arguments.done":
                        if current_tool_call:
                            if current_tool_args_parts:
                                current_tool_call["arguments"] = "".join(current_tool_args_parts)
                            elif event.get("arguments"):
                                current_tool_call["arguments"] = event.get("arguments")
                            _track_tool_call(current_tool_call)
                            current_tool_call = None
                            current_tool_args_parts = []
                        continue

                    if event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id", item.get("id", ""))
                            name = item.get("name", "")
                            initial_args = item.get("arguments", "") or ""
                            current_tool_call = {
                                "id": call_id,
                                "call_id": call_id,
                                "name": name,
                                "function_name": name,
                                "arguments": initial_args,
                            }
                            current_tool_args_parts = [initial_args] if initial_args else []
                        continue

                    if event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id", item.get("id", ""))
                            name = item.get("name", "")
                            arguments = item.get("arguments", _EMPTY_JSON)
                            _track_tool_call(
                                {
                                    "id": call_id,
                                    "call_id": call_id,
                                    "name": name,
                                    "function_name": name,
                                    "arguments": arguments,
                                }
                            )
                            current_tool_call = None
                            current_tool_args_parts = []
                        continue

                    if event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            completed_response = response_payload
                            final_model = response_payload.get("model", final_model)
                            stop_reason = response_payload.get("status", stop_reason)
                            usage_obj = self._extract_usage(response_payload)
                            final_usage_data = {
                                "prompt_tokens": usage_obj.prompt_tokens,
                                "completion_tokens": usage_obj.completion_tokens,
                                "total_tokens": usage_obj.total_tokens,
                            }
                        continue
            finally:
                stream_ctx.__exit__(None, None, None)

            if completed_response is not None:
                text_content, thinking_content, completed_tool_calls, stop_reason = self._extract_response_content(
                    completed_response
                )
                usage_obj = self._extract_usage(completed_response)
                final_model = completed_response.get("model", final_model)
                tool_calls: List[Dict[str, Any]] = completed_tool_calls
                if not tool_calls:
                    tool_calls = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id

                if tool_calls:
                    tool_calls_obj = self._parse_tool_call_sync(tool_call=tool_calls)
                    yield ToolCallList(model_id=final_model, tool_calls=tool_calls_obj, usage=usage_obj)
                else:
                    yield LLMResponse(
                        model_id=final_model,
                        type="llm_response",
                        text=text_content if text_content is not None else ("".join(completion_parts) or None),
                        thinking=thinking_content
                        if thinking_content is not None
                        else ("".join(thinking_parts) or None),
                        usage=usage_obj,
                        stop_reason=stop_reason,
                    )
            else:
                usage_obj = LLMUsage(**final_usage_data)
                tool_calls = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id
                if tool_calls:
                    tool_calls_obj = self._parse_tool_call_sync(tool_call=tool_calls)
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
            raise self._handle_error(e)

