import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union, Tuple

import anthropic

# Import specific exceptions from the anthropic library for better error handling
from anthropic import (APIConnectionError,
                       APIStatusError,
                       AuthenticationError,
                       BadRequestError,
                       InternalServerError,
                       NotFoundError,
                       PermissionDeniedError,
                       RateLimitError)

# Internal imports
from .base_provider import AsyncBaseProvider, BaseProvider
from .errors import (ConnectionError,
                     ContentFilterError,
                     ContextLengthExceededError,
                     EnvironmentError,
                     InvalidRequestError,
                     LLMError,
                     ModelNotFoundError,
                     QuotaExceededError,
                     ServerError)

from seqoria_agent.models.chunk import (AssistantMessage,
                                        SystemMessage,
                                        UserMessage,
                                        ToolCallRequest,
                                        ToolCallResult)

# Use a relative import since we're in the same directory
from .schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList

# Use optimized JSON utilities
from seqoria_agent.client.ai_client.utils import json_loads

_EMPTY_JSON: str = "{}"


class _AnthropicProviderMixin:
    """
    Shared functionality between sync and async Anthropic providers.
    
    This mixin contains common methods that don't depend on sync/async behavior.
    """
    
    def _init_common(self, model_id: str, api_key: Optional[str], **kwargs) -> tuple:
        """
        Common initialization logic shared between sync and async providers.
        
        Returns:
            tuple: (api_key, client_kwargs, model_id, default_params, max_retries)
        """
        # Prefer the explicitly supplied API key, otherwise fall back to the environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise EnvironmentError(message="Anthropic API key not provided", missing_vars=["ANTHROPIC_API_KEY"])
        
        # Store max_retries for use in retry logic
        max_retries = kwargs.get('max_retries', 3)
        
        # Client configuration
        client_kwargs = {'timeout': kwargs.get('timeout', 30.0),
                         'max_retries': 0}  # Disable automatic retries in the client; handled manually
        
        # model_id is required
        if not model_id:
            raise InvalidRequestError(message="model_id is required for Anthropic provider")
        
        effective_model_id = model_id

        # Initialize empty default_params - no hardcoded defaults
        default_params = {}
        
        # Define all valid parameter keys that can be passed to the Anthropic Messages API
        # See: https://docs.anthropic.com/en/api/messages
        valid_query_keys = {
            # Generation parameters
            "max_tokens", "temperature", "top_p", "top_k",
            "stop_sequences",
            # Extended thinking configuration
            # Can be: {"type": "enabled", "budget_tokens": 1024} or {"type": "disabled"}
            "thinking",
            # Service tier: "auto" or "standard_only"
            "service_tier",
            # Metadata
            "metadata",
        }
        
        # Only store user-provided kwargs that are valid
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value
        
        return api_key, client_kwargs, effective_model_id, default_params, max_retries
            
    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare request parameters by merging defaults with overrides.
        Removes parameters with None values.

        Args:
            **kwargs: Parameter overrides provided to generate/stream methods.
            
        Returns
        -------
            Dictionary of parameters ready for the Anthropic API request.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def _handle_error(self, error: Exception) -> Exception:
        """
        Map Anthropic errors to our standard error types using isinstance checks.
        
        Args:
            error: Original error from Anthropic client.
            
        Returns
        -------
            Mapped error instance inheriting from LLMError.
        """
        # Check specific Anthropic error types first
        if isinstance(error, AuthenticationError):
            return EnvironmentError(message=f"Anthropic authentication failed: {error}", missing_vars=["ANTHROPIC_API_KEY"])
        elif isinstance(error, RateLimitError):
            return QuotaExceededError(provider="Anthropic", limit_type="rate")
        elif isinstance(error, BadRequestError):
            error_message = str(object=error)
            if "context_length_exceeded" in error_message or "maximum context length" in error_message.lower():
                 tokens = max_tokens = None 
                 try:
                    parts = error_message.split(sep=' ')
                    for i, part in enumerate(iterable=parts):
                        if part == 'tokens': 
                            tokens = int(parts[i-1])
                        if part == 'limit': 
                            max_tokens = int(parts[i-1])
                 except (ValueError, IndexError):
                     pass
                 return ContextLengthExceededError(model_id=self.model_id, tokens=tokens, max_tokens=max_tokens)
            elif "content_policy_violation" in error_message or "safety" in error_message.lower():
                 return ContentFilterError(provider="Anthropic", filter_reason=error_message)
            elif "tool" in error_message.lower() and ("invalid" in error_message.lower() or "missing" in error_message.lower()):
                 # Handle tool-related validation errors
                 return InvalidRequestError(message=f"Tool validation error: {error}")
            elif "tool_choice" in error_message.lower():
                 # Handle tool_choice configuration errors
                 return InvalidRequestError(message=f"Tool choice configuration error: {error}")
            elif "input_schema" in error_message.lower():
                 # Handle tool schema validation errors
                 return InvalidRequestError(message=f"Tool schema validation error: {error}")
            else:
                 return InvalidRequestError(message=f"Invalid request to Anthropic: {error}")
        elif isinstance(error, NotFoundError):
             return ModelNotFoundError(model_id=self.model_id, provider="Anthropic")
        elif isinstance(error, APIConnectionError):
             return ConnectionError(provider="Anthropic", message=f"Connection error: {error}")
        elif isinstance(error, InternalServerError):
             return ServerError(provider="Anthropic", status_code=getattr(error, 'status_code', 500))
        elif isinstance(error, PermissionDeniedError):
             return InvalidRequestError(message=f"Anthropic permission denied: {error}")
        elif isinstance(error, APIStatusError):
            if 500 <= error.status_code < 600:
                return ServerError(provider="Anthropic", status_code=error.status_code)
            elif error.status_code == 429:
                return QuotaExceededError(provider="Anthropic", limit_type="rate")
            elif error.status_code == 400:
                 return InvalidRequestError(message=f"API Error {error.status_code}: {error}")
            else:
                 return LLMError(message=f"Anthropic API Error {error.status_code}: {error}")
        elif isinstance(error, asyncio.TimeoutError) or "timed out" in str(object=error).lower():
            return ConnectionError(provider="Anthropic", message="Request timed out")
            
        # Fallback for unexpected errors
        return LLMError(message=f"An unexpected Anthropic error occurred: {error}")
    
    def _parse_tool_call_sync(self, tool_call: List[Any]) -> List[ToolCall]:
        """Convert tool_call objects into ToolCall schema objects.

        Handles both dict-style and object-style tool calls from Anthropic responses.
        Ensures the arguments field is a mapping (JSON-decoded if it was
        received as a JSON string) so it satisfies the schema.
        """
        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            # Handle ToolCall objects that already have the right structure
            if isinstance(tc, ToolCall):
                processed_tool_calls.append(tc)
                continue
            
            # Handle dict-style tool calls
            if isinstance(tc, dict):
                args = tc.get("arguments", tc.get("input", {}))
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except Exception:
                        args = {}
                
                processed_tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    type="function_call",
                    function_name=tc.get("function_name", tc.get("name", "")),
                    arguments=args
                ))
            else:
                # Handle object-style (from Anthropic SDK)
                args = getattr(tc, 'arguments', getattr(tc, 'input', {}))
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except Exception:
                        args = {}
                
                processed_tool_calls.append(ToolCall(
                    id=getattr(tc, 'id', ''),
                    type="function_call",
                    function_name=getattr(tc, 'function_name', getattr(tc, 'name', '')),
                    arguments=args
                ))

        return processed_tool_calls
    
    def _format_tool_choice(self, tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        Format tool_choice parameter for the Anthropic Messages API.
        
        Supported formats:
            - String: "auto", "any", "none" -> {"type": "<value>"}
            - Dict with type: {"type": "auto"}, {"type": "any"}, {"type": "none"}
            - Dict with name: {"type": "tool", "name": "tool_name"}
            - Dict with disable_parallel_tool_use: {"type": "auto", "disable_parallel_tool_use": true}
        
        Args:
            tool_choice: Tool choice specification.
            
        Returns:
            Formatted tool_choice dict or None.
        """
        if tool_choice is None:
            return None
        
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "any", "none"}:
                return {"type": tool_choice}
            else:
                # Assume it's a tool name
                return {"type": "tool", "name": tool_choice}
        
        if isinstance(tool_choice, dict):
            # Ensure type is present
            if "type" not in tool_choice:
                if "name" in tool_choice:
                    return {"type": "tool", **tool_choice}
                else:
                    return {"type": "auto", **tool_choice}
            return tool_choice
        
        return None

    def _format_tool_call_sync(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the tool calls into the format expected by the Anthropic Messages API.
        
        Supports both OpenAI-style format and native Anthropic format:
        
        OpenAI format (converted):
            [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
        
        Anthropic native format (passed through):
            [{"name": "...", "description": "...", "input_schema": {...}}]
        
        Special tool types (passed through):
            - bash_20250124: {"type": "bash_20250124", "name": "bash"}
            - text_editor_20250124: {"type": "text_editor_20250124", "name": "str_replace_editor"}
            - web_search_20250305: {"type": "web_search_20250305", "name": "web_search", ...}
        
        Returns:
            List of tools in Anthropic format.
        """
        formatted_tools: List[Dict[str, Any]] = []
        
        for tool in tools:
            # Check if it's a special Anthropic tool type
            if "type" in tool and tool["type"] in {
                "bash_20250124", 
                "text_editor_20250124", 
                "text_editor_20250429",
                "text_editor_20250728",
                "web_search_20250305"
            }:
                # Pass through special tool types as-is
                formatted_tools.append(tool)
            elif "function" in tool:
                # OpenAI-style format - convert to Anthropic format
                func = tool["function"]
                formatted_tool = {
                    "name": str(func.get("name", "")),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                }
                # Add cache_control if provided
                if "cache_control" in tool:
                    formatted_tool["cache_control"] = tool["cache_control"]
                formatted_tools.append(formatted_tool)
            elif "name" in tool and "input_schema" in tool:
                # Already in Anthropic format - pass through
                formatted_tools.append(tool)
            elif "name" in tool:
                # Minimal format - add required fields
                formatted_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", tool.get("parameters", {"type": "object", "properties": {}}))
                })
        
        return formatted_tools

    def _convert_image_content_anthropic(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert image content blocks to Anthropic's vision format.
        
        Supported input formats:
            - {"type": "text", "text": "..."} - Text content (passed through)
            - {"type": "image", "url": "https://..."} - Image from URL
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"} - Base64 encoded image
            - {"type": "image", "source": {...}} - Already in Anthropic format (passed through)
        
        Returns:
            List of content blocks in Anthropic's format.
        """
        converted: List[Dict[str, Any]] = []
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                converted.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                # Check if already in Anthropic format
                if "source" in block:
                    converted.append(block)
                elif "url" in block:
                    # Convert URL format to Anthropic's format
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": block["url"]
                        }
                    })
                elif "base64" in block:
                    # Convert base64 format to Anthropic's format
                    media_type = block.get("media_type", "image/jpeg")
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": block["base64"]
                        }
                    })
            elif block_type == "image_url":
                # Convert OpenAI format to Anthropic format
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    # Parse data URL: data:image/jpeg;base64,{data}
                    try:
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        converted.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        })
                    except (ValueError, IndexError):
                        # Fallback if parsing fails
                        converted.append({
                            "type": "image",
                            "source": {"type": "url", "url": url}
                        })
                else:
                    converted.append({
                        "type": "image",
                        "source": {"type": "url", "url": url}
                    })
            else:
                # Unknown type, pass through as-is
                converted.append(block)
        
        return converted
    
    def _prepare_chat_history_sync(self,
                                   chat_history: List[Dict[str, Any]],
                                   *,
                                   enable_cache: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare the chat history for the Anthropic Messages API.

        Parameters
        ----------
        chat_history : List[Dict[str, Any]]
            Raw chat history provided by the caller (can include dataclass based
            messages as well as plain ``dict``s).
        enable_cache : bool, optional
            When *True*, the method will tag **system** blocks with
            ``{"cache_control": {"type": "ephemeral"}}`` so that the Anthropic
            backend will attempt to reuse these blocks across subsequent calls
            (see Anthropic prompt-caching documentation). Defaults to *False* so
            existing behaviour is preserved.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
            Two lists representing the formatted ``system`` blocks and the
            conversation ``messages`` respectively, ready to be forwarded to
            ``anthropic.messages.create``.
        
        Message Format (per Anthropic API spec):
            - System messages are extracted to a separate `system` parameter
            - Messages array must alternate between user and assistant roles
            - Tool calls appear as `tool_use` blocks in assistant messages
            - Tool results appear as `tool_result` blocks in user messages
        
        Supports vision/image content in messages. Images can be included using:
            - {"type": "image", "url": "https://..."} - Image from URL (URLImageSource)
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"} - Base64 encoded (Base64ImageSource)
        
        Example with images:
            [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "url": "https://example.com/image.jpg"}
                ]}
            ]
        """
        prepared_chat_history: List[Dict[str, Any]] = []
        system_message: List[Dict[str, Any]] = []

        for message in chat_history:
            # Handle ToolCallRequest pydantic model
            # This represents the assistant requesting to use a tool (tool_use block)
            if isinstance(message, ToolCallRequest):
                # Parse arguments - ensure it's a dict
                args = message.arguments
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except (ValueError, TypeError):
                        args = {}
                elif args is None:
                    args = {}
                
                # Tool use goes in assistant message content
                prepared_chat_history.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": str(message.id),
                        "name": message.name,
                        "input": args
                    }]
                })
                continue

            # Handle ToolCallResult pydantic model
            # This represents the result of a tool execution (tool_result block)
            if isinstance(message, ToolCallResult):
                # Tool result goes in user message content
                result_content = message.result
                # Content can be string or list of content blocks
                if isinstance(result_content, str):
                    content_value = result_content
                else:
                    content_value = str(result_content)
                
                prepared_chat_history.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": str(message.tool_call_id),
                        "content": content_value
                    }]
                })
                continue

            # Handle AssistantMessage and UserMessage pydantic models
            if isinstance(message, (AssistantMessage, UserMessage)):
                content = message.content
                # Check if content is multimodal (list of content blocks)
                if isinstance(content, list):
                    content = self._convert_image_content_anthropic(content)
                    prepared_chat_history.append({"role": message.role, "content": content})
                else:
                    # Convert string to text content block
                    prepared_chat_history.append({
                        "role": message.role,
                        "content": [{"type": "text", "text": str(content)}]
                    })
                continue

            # Handle SystemMessage pydantic model
            if isinstance(message, SystemMessage):
                sys_block = {"type": "text", "text": message.content}
                if enable_cache:
                    sys_block["cache_control"] = {"type": "ephemeral"}
                system_message.append(sys_block)
                continue

            # Handle dict messages
            role = message.get("role")
            
            # Handle system messages (goes to separate system parameter)
            if role == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    sys_block = {"type": "text", "text": content}
                elif isinstance(content, list):
                    # Already in content block format
                    for block in content:
                        sys_block = block.copy()
                        if enable_cache:
                            sys_block["cache_control"] = {"type": "ephemeral"}
                        system_message.append(sys_block)
                    continue
                else:
                    sys_block = {"type": "text", "text": str(content)}
                
                if enable_cache:
                    sys_block["cache_control"] = {"type": "ephemeral"}
                system_message.append(sys_block)
                
            # Handle tool_call (internal format for tool use request)
            elif role == "tool_call":
                args = message.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except (ValueError, TypeError):
                        args = {}
                
                prepared_chat_history.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": str(message.get("id", "")),
                        "name": message.get("name", ""),
                        "input": args
                    }]
                })
                
            # Handle tool_result (internal format for tool execution result)
            elif role == "tool_result" or role == "tool":
                result = message.get("result", message.get("content", ""))
                tool_use_id = message.get("tool_call_id", message.get("tool_use_id", ""))
                is_error = message.get("is_error", False)
                
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": str(tool_use_id),
                    "content": str(result) if isinstance(result, str) else result
                }
                
                if is_error:
                    tool_result_block["is_error"] = True
                
                prepared_chat_history.append({
                    "role": "user",
                    "content": [tool_result_block]
                })
                
            # Handle regular assistant/user messages
            elif role in {"assistant", "user"}:
                content = message.get("content")
                
                # Check if content is multimodal (list of content blocks)
                if isinstance(content, list):
                    # Process content blocks - convert images and handle special types
                    processed_content = []
                    for block in content:
                        block_type = block.get("type")
                        
                        # Handle thinking blocks (for multi-turn with extended thinking)
                        if block_type == "thinking":
                            processed_content.append({
                                "type": "thinking",
                                "thinking": block.get("thinking", ""),
                                "signature": block.get("signature", "")
                            })
                        # Handle redacted thinking blocks
                        elif block_type == "redacted_thinking":
                            processed_content.append({
                                "type": "redacted_thinking",
                                "data": block.get("data", "")
                            })
                        # Handle tool_use blocks in assistant messages
                        elif block_type == "tool_use":
                            args = block.get("input", {})
                            if isinstance(args, str):
                                try:
                                    args = json_loads(args)
                                except (ValueError, TypeError):
                                    args = {}
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": args
                            })
                        # Handle tool_result blocks in user messages
                        elif block_type == "tool_result":
                            processed_content.append({
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                                **({"is_error": True} if block.get("is_error") else {})
                            })
                        else:
                            # Convert images and pass through other types
                            converted = self._convert_image_content_anthropic([block])
                            processed_content.extend(converted)
                    
                    prepared_chat_history.append({"role": role, "content": processed_content})
                else:
                    # String content - wrap in text block
                    prepared_chat_history.append({
                        "role": role,
                        "content": [{"type": "text", "text": str(content) if content else ""}]
                    })

        # Apply cache control to first message if no system message and caching enabled
        if enable_cache and not system_message and prepared_chat_history:
            first_msg = prepared_chat_history[0]
            content = first_msg.get("content")
            if isinstance(content, list) and content:
                # Attach marker to the final block of the list
                content[-1]["cache_control"] = {"type": "ephemeral"}

        return system_message, prepared_chat_history
    
    def _validate_model_sync(self, client: "anthropic.Anthropic", model_id: str) -> None:
        """
        Validate that the requested model is available via the Anthropic API.
        
        Args:
            client: The Anthropic client instance.
            model_id: The model identifier to validate.
            
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        try:
            models_response = client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Anthropic",
                    available_models=available_models
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            # If we can't fetch models, wrap the error
            mapped_error = self._handle_error(error=e)
            raise mapped_error
    
    async def _validate_model_async(self, async_client: "anthropic.AsyncAnthropic", model_id: str) -> None:
        """
        Asynchronously validate that the requested model is available via the Anthropic API.
        
        Args:
            async_client: The async Anthropic client instance.
            model_id: The model identifier to validate.
            
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        try:
            models_response = await async_client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if model_id not in available_models:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Anthropic",
                    available_models=available_models
                )
        except ModelNotFoundError:
            raise
        except Exception as e:
            # If we can't fetch models, wrap the error
            mapped_error = self._handle_error(error=e)
            raise mapped_error


class AsyncAnthropicProvider(AsyncBaseProvider, _AnthropicProviderMixin):
    """
    Asynchronous provider implementation for Anthropic's Claude API.
    
    This class provides async methods to generate text completions
    and stream responses from Anthropic's Claude models,
    integrating robust error handling and retry mechanisms.
    Supports tool usage and function calling.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the asynchronous Anthropic provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"). Required.
            api_key: Explicit API key for authentication. If None, falls back to ANTHROPIC_API_KEY environment variable.
            **kwargs: Additional parameters for model generation and client configuration:
                Query kwargs (passed to Anthropic API):
                    - temperature (float): Sampling temperature (default: 0.7)
                    - max_tokens (int): Maximum tokens to generate (default: 1024).
                    - top_p (float): Nucleus sampling parameter (default: 1.0)
                    - top_k (int): Top-k sampling parameter
                    - stop_sequences (List[str]): Sequence(s) that stop generation.
                    - metadata (dict): Metadata for the request
                    - tools (List[Dict]): List of tools available for the model to use
                    - tool_choice (Dict): Tool choice configuration
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
        
        Raises
        ------
            EnvironmentError: If ANTHROPIC_API_KEY environment variable is not set.
            ConnectionError: If the async client cannot be initialized.
        """
        api_key, client_kwargs, self.model_id, self.default_params, self.max_retries = self._init_common(
            model_id, api_key, **kwargs
        )
        
        try:
            # Create and assign the async client
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="Anthropic") from e
    
    async def validate_model(self) -> None:
        """
        Validate that the configured model is available via the Anthropic API.
        
        Call this method after initialization to verify the model exists.
        
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        
        Example:
            provider = AsyncAnthropicProvider(model_id="claude-sonnet-4-20250514", api_key="...")
            await provider.validate_model()  # Raises if model not found
        """
        await self._validate_model_async(async_client=self.async_client, model_id=self.model_id)
    
    # Reuse shared implementation from AsyncBaseProvider
    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)
    
    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        """Convert *tool_call* dictionaries into :class:`ToolCall` objects."""
        return self._parse_tool_call_sync(tool_call)
    
    async def _format_tool_call(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format the tool calls into the format expected by the Anthropic API."""
        return self._format_tool_call_sync(tools)

    async def _prepare_chat_history(self,
                                    chat_history: List[Dict[str, Any]],
                                    *,
                                    enable_cache: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare the chat history for the Anthropic API."""
        return self._prepare_chat_history_sync(chat_history, enable_cache=enable_cache)

    async def async_generate(self,
                             chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                             tools: Optional[List[Dict[str, Any]]] = None,
                             tool_choice: Optional[Dict[str, Any]] = None,
                             model: Optional[str] = None,
                             **kwargs) -> Union[LLMResponse, ToolCallList]:
        """
        Asynchronously generate a completion for the provided chat history.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Image formats supported:
            - {"type": "image", "url": "https://..."} - Image from URL
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"} - Base64 encoded
        
        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: Optional list of tool definitions for function calling.
                   Each tool should have 'name', 'description', and 'input_schema' fields.
            tool_choice: Optional tool choice configuration. Can be:
                        - {"type": "auto"} (default): Let Claude decide whether to use tools
                        - {"type": "any"}: Force Claude to use at least one tool
                        - {"type": "tool", "name": "tool_name"}: Force Claude to use a specific tool
            model: Optional model identifier to override the default model
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).
            
        Returns
        -------
            Union[LLMResponse, ToolCallList]: Response containing:
                - text (str | None): Generated text string. None if generation fails or is empty.
                - usage (LLMUsage): Token usage information
                - tool_calls (List[ToolCall] | None): List of tool calls made by the model, if any.
                - stop_reason (str): Reason why generation stopped ("end_turn", "tool_use", etc.)
            
        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = await self._format_tool_call(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        if chat_history:
            system_message, formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        async def _generate() -> Union[LLMResponse, ToolCallList]:
            
            completion = await self.async_client.messages.create(model=effective_model,
                                                                 messages=formatted_chat_history,
                                                                 system=system_message,
                                                                 **params)
            
            # Extract text content, thinking, and tool calls
            text_parts: List[str] = []
            thinking_parts: List[str] = []
            tool_calls = []
            
            if completion.content and len(completion.content) > 0:
                for content_block in completion.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'text' and hasattr(content_block, 'text'):
                            if content_block.text:
                                text_parts.append(content_block.text)
                        elif content_block.type == 'thinking' and hasattr(content_block, 'thinking'):
                            # Extract thinking/reasoning content (Claude extended thinking)
                            if content_block.thinking:
                                thinking_parts.append(content_block.thinking)
                        elif content_block.type == 'tool_use':
                            try:
                                parsed_args = json_loads(content_block.input) if isinstance(content_block.input, str) else content_block.input
                            except Exception:
                                parsed_args = content_block.input

                            # Extract tool use information
                            tool_calls.append(ToolCall(id=content_block.id,
                                                       type="function_call",
                                                       function_name=content_block.name,
                                                       arguments=parsed_args))

            usage_data = LLMUsage(prompt_tokens=0, 
                                  completion_tokens=0, 
                                  total_tokens=0)
            
            if completion.usage:
                usage_data.prompt_tokens = completion.usage.input_tokens or 0
                usage_data.completion_tokens = completion.usage.output_tokens or 0
                usage_data.total_tokens = (completion.usage.input_tokens or 0) + (completion.usage.output_tokens or 0)

            if tool_calls:
                return ToolCallList(model_id=effective_model,
                                    tool_calls=tool_calls,
                                    usage=usage_data)

            text_content = "".join(text_parts) if text_parts else None
            thinking_content = "".join(thinking_parts) if thinking_parts else None

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=text_content,
                               thinking=thinking_content,
                               usage=usage_data,
                               stop_reason=completion.stop_reason)
        
        return await self._execute_with_retry(operation=_generate)
    
    async def async_stream(self,
                           chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                           tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Dict[str, Any]] = None,
                           model: Optional[str] = None,
                           **kwargs) -> AsyncGenerator[Union[str, LLMResponse, ToolCallList], None]:
        """
        Asynchronously stream a completion for the provided chat history.
        
        Yields content chunks (str) as they arrive, and finally yields a
        LLMResponse containing the full concatenated text, tool calls, and token usage.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Args:
            chat_history: List of message dictionaries.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration.
            model: Optional model identifier to override the default model
            **kwargs: Generation parameters overriding defaults.
            
        Yields
        ------
            str: Text chunks of the generated completion.
            LLMResponse: Final LLMResponse with "text" (full content), "usage" info, "tool_calls", and "stop_reason".
                Example: {
                    "text": "Full response", 
                    "usage": {...}, 
                    "tool_calls": [...],
                    "stop_reason": "tool_use"
                }

        Raises
        ------
            Same exceptions as async_generate() method, potentially during iteration.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = await self._format_tool_call(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        if chat_history:
            system_message, formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        async def _create_stream() -> Any:
            # Use the async client here
            return await self.async_client.messages.create(model=effective_model,
                                                          messages=formatted_chat_history,
                                                          system=system_message,
                                                          stream=True,
                                                          **params)
        
        completion_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        current_tool_call: ToolCall | None = None
        current_tool_input_parts: List[str] = []
        current_block_type = None  # Track current content block type
        final_usage_data = LLMUsage(prompt_tokens=0, 
                                    completion_tokens=0, 
                                    total_tokens=0)
        stop_reason = None

        try:
            # Get the stream manager and iterate through chunks
            async for chunk in await self._execute_with_retry(operation=_create_stream):
                # Handle different event types from Anthropic streaming
                if chunk.type == "message_start":
                    # Extract usage from message start event
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                        final_usage_data.prompt_tokens = chunk.message.usage.input_tokens or 0
                elif chunk.type == "content_block_start":
                    # Handle start of content blocks (text, thinking, or tool_use)
                    if hasattr(chunk, 'content_block'):
                        current_block_type = chunk.content_block.type
                        if chunk.content_block.type == 'tool_use':
                            # Start of a tool use block
                            current_tool_call = ToolCall(id=chunk.content_block.id,
                                                         type="tool_use",
                                                         function_name=chunk.content_block.name,
                                                         arguments=chunk.content_block.input or {})  # May be empty initially
                            current_tool_input_parts = []  # Reset for accumulating JSON parts
                elif chunk.type == "content_block_delta":
                    # Extract text/thinking delta from content block delta
                    if hasattr(chunk, 'delta'):
                        if hasattr(chunk.delta, 'text'):
                            # Text content delta
                            delta_text = chunk.delta.text
                            if delta_text:
                                completion_parts.append(delta_text)
                                yield LLMResponse(model_id=effective_model, 
                                                  type="llm_delta", 
                                                  text=delta_text,
                                                  thinking=None,
                                                  usage=None,
                                                  stop_reason=None)
                        elif hasattr(chunk.delta, 'thinking'):
                            # Thinking content delta (Claude extended thinking)
                            delta_thinking = chunk.delta.thinking
                            if delta_thinking:
                                thinking_parts.append(delta_thinking)
                                yield LLMResponse(model_id=effective_model, 
                                                  type="llm_delta", 
                                                  text=None,
                                                  thinking=delta_thinking,
                                                  usage=None,
                                                  stop_reason=None)
                        elif hasattr(chunk.delta, 'partial_json') and current_tool_call:
                            # Tool use input delta (partial JSON)
                            # Accumulate the partial JSON strings
                            pj = chunk.delta.partial_json
                            if pj:
                                current_tool_input_parts.append(pj)
                elif chunk.type == "content_block_stop":
                    # End of a content block
                    if current_tool_call:
                        # Complete the tool call and add it to the list
                        try:
                            # Parse the accumulated JSON if we have it
                            if current_tool_input_parts:
                                current_tool_call.arguments = json_loads("".join(current_tool_input_parts))
                            tool_calls.append(current_tool_call)
                        except (ValueError, TypeError):
                            # If JSON parsing fails, use the input from the start block
                            tool_calls.append(current_tool_call)
                        
                        current_tool_call = None
                        current_tool_input_parts = []
                    current_block_type = None
                elif chunk.type == "message_delta":
                    # Extract final usage and stop reason from message delta event
                    if hasattr(chunk, 'usage'):
                        final_usage_data.completion_tokens = chunk.usage.output_tokens or 0
                        final_usage_data.total_tokens = final_usage_data.prompt_tokens + final_usage_data.completion_tokens
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                        stop_reason = chunk.delta.stop_reason
            

            full_completion_text = "".join(completion_parts)
            full_thinking_text = "".join(thinking_parts)

            if tool_calls:
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls,
                                   usage=final_usage_data)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text,
                                  thinking=full_thinking_text if full_thinking_text else None,
                                  usage=final_usage_data,
                                  stop_reason=stop_reason)

        except Exception as e:
            # Handle errors occurring during async stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error


class AnthropicProvider(BaseProvider, _AnthropicProviderMixin):
    """
    Synchronous provider implementation for Anthropic's Claude API.
    
    This class provides sync methods to generate text completions
    and stream responses from Anthropic's Claude models,
    integrating robust error handling and retry mechanisms.
    Supports tool usage and function calling.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the synchronous Anthropic provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"). Required.
            api_key: Explicit API key for authentication. If None, falls back to ANTHROPIC_API_KEY environment variable.
            **kwargs: Additional parameters for model generation and client configuration:
                Query kwargs (passed to Anthropic API):
                    - temperature (float): Sampling temperature (default: 0.7)
                    - max_tokens (int): Maximum tokens to generate (default: 1024).
                    - top_p (float): Nucleus sampling parameter (default: 1.0)
                    - top_k (int): Top-k sampling parameter
                    - stop_sequences (List[str]): Sequence(s) that stop generation.
                    - metadata (dict): Metadata for the request
                    - tools (List[Dict]): List of tools available for the model to use
                    - tool_choice (Dict): Tool choice configuration
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
        
        Raises
        ------
            EnvironmentError: If ANTHROPIC_API_KEY environment variable is not set.
            ConnectionError: If the sync client cannot be initialized.
        """
        api_key, client_kwargs, self.model_id, self.default_params, self.max_retries = self._init_common(
            model_id, api_key, **kwargs
        )
        
        try:
            # Create and assign the sync client
            self.client = anthropic.Anthropic(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="Anthropic") from e
        
        # Validate that the requested model is available
        self._validate_model_sync(client=self.client, model_id=self.model_id)
    
    def validate_model(self) -> None:
        """
        Validate that the configured model is available via the Anthropic API.
        
        This is called automatically during initialization, but can be called
        again if needed (e.g., after changing model_id).
        
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        self._validate_model_sync(client=self.client, model_id=self.model_id)

    def generate(self,
                 chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 tool_choice: Optional[Dict[str, Any]] = None,
                 model: Optional[str] = None,
                 **kwargs) -> Union[LLMResponse, ToolCallList]:
        """
        Synchronously generate a completion for the provided chat history.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Image formats supported:
            - {"type": "image", "url": "https://..."} - Image from URL
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"} - Base64 encoded
        
        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: Optional list of tool definitions for function calling.
                   Each tool should have 'name', 'description', and 'input_schema' fields.
            tool_choice: Optional tool choice configuration. Can be:
                        - {"type": "auto"} (default): Let Claude decide whether to use tools
                        - {"type": "any"}: Force Claude to use at least one tool
                        - {"type": "tool", "name": "tool_name"}: Force Claude to use a specific tool
            model: Optional model identifier to override the default model
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).
            
        Returns
        -------
            Union[LLMResponse, ToolCallList]: Response containing:
                - text (str | None): Generated text string. None if generation fails or is empty.
                - usage (LLMUsage): Token usage information
                - tool_calls (List[ToolCall] | None): List of tool calls made by the model, if any.
                - stop_reason (str): Reason why generation stopped ("end_turn", "tool_use", etc.)
            
        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = self._format_tool_call_sync(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        if chat_history:
            system_message, formatted_chat_history = self._prepare_chat_history_sync(chat_history=chat_history)

        effective_model = model or self.model_id
        
        def _generate() -> Union[LLMResponse, ToolCallList]:
            completion = self.client.messages.create(model=effective_model,
                                                     messages=formatted_chat_history,
                                                     system=system_message,
                                                     **params)
            
            # Extract text content, thinking, and tool calls
            text_parts: List[str] = []
            thinking_parts: List[str] = []
            tool_calls = []
            
            if completion.content and len(completion.content) > 0:
                for content_block in completion.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'text' and hasattr(content_block, 'text'):
                            if content_block.text:
                                text_parts.append(content_block.text)
                        elif content_block.type == 'thinking' and hasattr(content_block, 'thinking'):
                            # Extract thinking/reasoning content (Claude extended thinking)
                            if content_block.thinking:
                                thinking_parts.append(content_block.thinking)
                        elif content_block.type == 'tool_use':
                            try:
                                parsed_args = json_loads(content_block.input) if isinstance(content_block.input, str) else content_block.input
                            except Exception:
                                parsed_args = content_block.input

                            tool_calls.append(ToolCall(id=content_block.id,
                                                       type="function_call",
                                                       function_name=content_block.name,
                                                       arguments=parsed_args))

            usage_data = LLMUsage(prompt_tokens=0, 
                                  completion_tokens=0, 
                                  total_tokens=0)
            
            if completion.usage:
                usage_data.prompt_tokens = completion.usage.input_tokens or 0
                usage_data.completion_tokens = completion.usage.output_tokens or 0
                usage_data.total_tokens = (completion.usage.input_tokens or 0) + (completion.usage.output_tokens or 0)

            if tool_calls:
                return ToolCallList(model_id=effective_model,
                                    tool_calls=tool_calls,
                                    usage=usage_data)

            text_content = "".join(text_parts) if text_parts else None
            thinking_content = "".join(thinking_parts) if thinking_parts else None

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=text_content,
                               thinking=thinking_content,
                               usage=usage_data,
                               stop_reason=completion.stop_reason)
        
        return self._execute_with_retry(operation=_generate)
    
    def stream(self,
               chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
               tools: Optional[List[Dict[str, Any]]] = None,
               tool_choice: Optional[Dict[str, Any]] = None,
               model: Optional[str] = None,
               **kwargs) -> Generator[Union[str, LLMResponse, ToolCallList], None, None]:
        """
        Synchronously stream a completion for the provided chat history.
        
        Yields content chunks as they arrive, and finally yields a
        LLMResponse containing the full concatenated text, tool calls, and token usage.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Args:
            chat_history: List of message dictionaries.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration.
            model: Optional model identifier to override the default model
            **kwargs: Generation parameters overriding defaults.
            
        Yields
        ------
            LLMResponse: Delta chunks and final response with full content, usage info, and tool_calls.

        Raises
        ------
            Same exceptions as generate() method, potentially during iteration.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = self._format_tool_call_sync(tools=tools)
        if tool_choice is not None:
            params["tool_choice"] = self._format_tool_choice(tool_choice)

        if chat_history:
            system_message, formatted_chat_history = self._prepare_chat_history_sync(chat_history=chat_history)

        effective_model = model or self.model_id
        
        def _create_stream() -> Any:
            return self.client.messages.create(model=effective_model,
                                               messages=formatted_chat_history,
                                               system=system_message,
                                               stream=True,
                                               **params)
        
        completion_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        current_tool_call: ToolCall | None = None
        current_tool_input_parts: List[str] = []
        current_block_type = None  # Track current content block type
        final_usage_data = LLMUsage(prompt_tokens=0, 
                                    completion_tokens=0, 
                                    total_tokens=0)
        stop_reason = None

        try:
            # Get the stream and iterate through chunks
            for chunk in self._execute_with_retry(operation=_create_stream):
                # Handle different event types from Anthropic streaming
                if chunk.type == "message_start":
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                        final_usage_data.prompt_tokens = chunk.message.usage.input_tokens or 0
                elif chunk.type == "content_block_start":
                    if hasattr(chunk, 'content_block'):
                        current_block_type = chunk.content_block.type
                        if chunk.content_block.type == 'tool_use':
                            current_tool_call = ToolCall(id=chunk.content_block.id,
                                                         type="tool_use",
                                                         function_name=chunk.content_block.name,
                                                         arguments=chunk.content_block.input or {})
                            current_tool_input_parts = []
                elif chunk.type == "content_block_delta":
                    if hasattr(chunk, 'delta'):
                        if hasattr(chunk.delta, 'text'):
                            delta_text = chunk.delta.text
                            if delta_text:
                                completion_parts.append(delta_text)
                                yield LLMResponse(model_id=effective_model, 
                                                  type="llm_delta", 
                                                  text=delta_text,
                                                  thinking=None,
                                                  usage=None,
                                                  stop_reason=None)
                        elif hasattr(chunk.delta, 'thinking'):
                            # Thinking content delta (Claude extended thinking)
                            delta_thinking = chunk.delta.thinking
                            if delta_thinking:
                                thinking_parts.append(delta_thinking)
                                yield LLMResponse(model_id=effective_model, 
                                                  type="llm_delta", 
                                                  text=None,
                                                  thinking=delta_thinking,
                                                  usage=None,
                                                  stop_reason=None)
                        elif hasattr(chunk.delta, 'partial_json') and current_tool_call:
                            pj = chunk.delta.partial_json
                            if pj:
                                current_tool_input_parts.append(pj)
                elif chunk.type == "content_block_stop":
                    if current_tool_call:
                        try:
                            if current_tool_input_parts:
                                current_tool_call.arguments = json_loads("".join(current_tool_input_parts))
                            tool_calls.append(current_tool_call)
                        except (ValueError, TypeError):
                            tool_calls.append(current_tool_call)
                        
                        current_tool_call = None
                        current_tool_input_parts = []
                    current_block_type = None
                elif chunk.type == "message_delta":
                    if hasattr(chunk, 'usage'):
                        final_usage_data.completion_tokens = chunk.usage.output_tokens or 0
                        final_usage_data.total_tokens = final_usage_data.prompt_tokens + final_usage_data.completion_tokens
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                        stop_reason = chunk.delta.stop_reason
            

            full_completion_text = "".join(completion_parts)
            full_thinking_text = "".join(thinking_parts)

            if tool_calls:
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls,
                                   usage=final_usage_data)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text,
                                  thinking=full_thinking_text if full_thinking_text else None,
                                  usage=final_usage_data,
                                  stop_reason=stop_reason)

        except Exception as e:
            mapped_error = self._handle_error(error=e)
            raise mapped_error
