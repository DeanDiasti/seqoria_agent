"""
Ollama provider implementation for local and cloud LLM inference.

This module provides synchronous and asynchronous provider classes for interacting
with Ollama's API, supporting chat completions, streaming, tool calling, and vision.

Ollama Python SDK Documentation: https://github.com/ollama/ollama-python
"""

import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

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
from seqoria_agent.models.chunk import (
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ToolCallRequest,
    ToolCallResult,
)

_EMPTY_JSON: str = "{}"


class _OllamaProviderMixin:
    """
    Shared functionality between sync and async Ollama providers.
    
    This mixin contains common methods that don't depend on sync/async behavior.
    """
    
    def _init_common(
        self,
        model_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Common initialization logic shared between sync and async providers.
        
        Args:
            model_id: The model identifier (required).
            host: Ollama server URL (default: http://localhost:11434 or OLLAMA_HOST env).
            api_key: Optional API key for cloud Ollama (ollama.com).
            **kwargs: Additional parameters for generation.
        
        Returns:
            tuple: (host, headers, model_id, default_params, max_retries, auto_pull)
        """
        # model_id is required
        if not model_id:
            raise InvalidRequestError(message="model_id is required for Ollama provider")
        
        effective_model_id = model_id
        
        # Determine host - priority: explicit > env > default
        effective_host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Build headers if API key is provided (for cloud Ollama)
        headers: Dict[str, str] = {}
        effective_api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if effective_api_key:
            headers["Authorization"] = f"Bearer {effective_api_key}"
        
        # Store max_retries for use in retry logic
        max_retries = kwargs.get('max_retries', 3)
        
        # Auto-pull: attempt to pull model if not found
        auto_pull = kwargs.get('auto_pull', True)
        
        # Initialize empty default_params - no hardcoded defaults
        default_params: Dict[str, Any] = {}
        
        # Define all valid parameter keys that can be passed to the Ollama API
        # These map to Ollama's "options" and top-level request parameters
        valid_query_keys = {
            # Top-level params
            "format",
            "keep_alive",
            "think",
            # Options (generation params)
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "num_ctx",
            "num_predict",
            "logprobs",
            "top_logprobs",
        }
        
        # Only store user-provided kwargs that are valid
        for key, value in kwargs.items():
            if key in valid_query_keys:
                default_params[key] = value
        
        return effective_host, headers, effective_model_id, default_params, max_retries, auto_pull
    
    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare request parameters by merging defaults with overrides.
        Removes parameters with None values.
        
        Args:
            **kwargs: Parameter overrides provided to generate/stream methods.
        
        Returns:
            Dictionary of parameters ready for the Ollama API request.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        params = {k: v for k, v in params.items() if v is not None}
        return params
    
    def _build_options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the 'options' dict for Ollama API from params.
        Separates top-level params from generation options.
        
        Args:
            params: All parameters including both top-level and options.
        
        Returns:
            Tuple of (top_level_params, options_dict)
        """
        top_level_keys = {"format", "keep_alive", "think"}
        options_keys = {"temperature", "top_p", "top_k", "min_p", "seed", 
                        "stop", "num_ctx", "num_predict"}
        
        top_level = {}
        options = {}
        
        for key, value in params.items():
            if key in top_level_keys:
                top_level[key] = value
            elif key in options_keys:
                options[key] = value
        
        return top_level, options
    
    def _handle_error(self, error: Exception) -> Exception:
        """
        Map Ollama errors to our standard error types.
        
        Args:
            error: Original error from Ollama client.
        
        Returns:
            Mapped error instance inheriting from LLMError.
        """
        error_str = str(error).lower()
        
        # Import Ollama's ResponseError if available
        try:
            from ollama import ResponseError
            if isinstance(error, ResponseError):
                status_code = getattr(error, 'status_code', None)
                
                if status_code == 404:
                    return ModelNotFoundError(model_id=self.model_id, provider="Ollama")
                elif status_code == 429:
                    return QuotaExceededError(provider="Ollama", limit_type="rate")
                elif status_code and 500 <= status_code < 600:
                    return ServerError(provider="Ollama", status_code=status_code)
                elif status_code == 400:
                    return InvalidRequestError(message=f"Ollama bad request: {error}")
        except ImportError:
            pass
        
        # Check for common error patterns in the message
        if "connection" in error_str or "refused" in error_str or "unreachable" in error_str:
            return ConnectionError(provider="Ollama", message=f"Connection error: {error}")
        elif "timeout" in error_str or "timed out" in error_str:
            return ConnectionError(provider="Ollama", message="Request timed out")
        elif "context length" in error_str or "too long" in error_str:
            return ContextLengthExceededError(model_id=self.model_id, tokens=None, max_tokens=None)
        elif "not found" in error_str:
            return ModelNotFoundError(model_id=self.model_id, provider="Ollama")
        
        # Fallback for unexpected errors
        return LLMError(message=f"An unexpected Ollama error occurred: {error}")
    
    def _convert_image_content_ollama(self, content: Union[str, List[Dict[str, Any]]]) -> tuple:
        """
        Convert image content blocks to Ollama's format.
        
        Ollama expects images as base64 strings in a separate 'images' field.
        
        Supported input formats:
            - str: Plain text content (passed through)
            - [{"type": "text", "text": "..."}] - Text content blocks
            - [{"type": "image", "url": "..."}] - Image URL (will be fetched and converted)
            - [{"type": "image", "base64": "...", "media_type": "..."}] - Base64 image
            - [{"type": "image_url", "image_url": {"url": "..."}}] - OpenAI format
        
        Returns:
            Tuple of (text_content: str, images: List[str])
        """
        if isinstance(content, str):
            return content, []
        
        text_parts: List[str] = []
        images: List[str] = []
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "image":
                # Our common format
                if "base64" in block:
                    images.append(block["base64"])
                elif "url" in block:
                    url = block["url"]
                    # Check if it's a data URL
                    if url.startswith("data:"):
                        # Extract base64 from data URL
                        try:
                            base64_data = url.split(",", 1)[1]
                            images.append(base64_data)
                        except (IndexError, ValueError):
                            pass
                    else:
                        # For regular URLs, we'd need to fetch - skip for now
                        # Users should provide base64 for Ollama
                        text_parts.append(f"[Image URL: {url}]")
            elif block_type == "image_url":
                # OpenAI format
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    try:
                        base64_data = url.split(",", 1)[1]
                        images.append(base64_data)
                    except (IndexError, ValueError):
                        pass
                else:
                    text_parts.append(f"[Image URL: {url}]")
        
        return " ".join(text_parts), images
    
    def _prepare_chat_history_sync(
        self,
        chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare chat history for Ollama API.
        
        Supports vision/image content in messages. Images should be provided as base64:
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"}
        
        Example with images:
            [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "base64": "iVBORw0KGgo..."}
                ]}
            ]
        
        Returns:
            List of messages in Ollama's format.
        """
        prepared_messages: List[Dict[str, Any]] = []
        
        for raw_msg in chat_history:
            # Handle Pydantic model inputs first
            if isinstance(raw_msg, ToolCallRequest):
                # Ollama expects tool calls in assistant messages
                prepared_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": raw_msg.name,
                            "arguments": raw_msg.arguments or {}
                        }
                    }]
                })
                continue
            
            if isinstance(raw_msg, ToolCallResult):
                prepared_messages.append({
                    "role": "tool",
                    "content": str(raw_msg.result)
                })
                continue
            
            if isinstance(raw_msg, (AssistantMessage, UserMessage, SystemMessage)):
                content = raw_msg.content
                text_content, images = self._convert_image_content_ollama(content)
                msg = {"role": raw_msg.role, "content": text_content}
                if images:
                    msg["images"] = images
                prepared_messages.append(msg)
                continue
            
            # Handle dict messages
            role = raw_msg.get("role")
            content = raw_msg.get("content", "")
            
            if role == "tool_call":
                prepared_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": raw_msg.get("name"),
                            "arguments": raw_msg.get("arguments", {})
                        }
                    }]
                })
            elif role == "tool_result":
                prepared_messages.append({
                    "role": "tool",
                    "content": str(raw_msg.get("result", ""))
                })
            elif role in {"assistant", "system", "user"}:
                text_content, images = self._convert_image_content_ollama(content)
                msg = {"role": role, "content": text_content}
                if images:
                    msg["images"] = images
                prepared_messages.append(msg)
        
        return prepared_messages
    
    async def _prepare_chat_history_async(
        self,
        chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]]
    ) -> List[Dict[str, Any]]:
        """Async version - delegates to sync implementation."""
        return self._prepare_chat_history_sync(chat_history)
    
    def _convert_tools_for_ollama(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Convert tool definitions to Ollama's format.
        
        Ollama uses a format similar to OpenAI's function calling.
        
        Args:
            tools: List of tool definitions with 'name', 'description', 'input_schema'.
        
        Returns:
            List of tools in Ollama's format, or None if no tools.
        """
        if not tools:
            return None
        
        ollama_tools = []
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {}))
                }
            }
            ollama_tools.append(ollama_tool)
        
        return ollama_tools
    
    def _parse_tool_calls(self, tool_calls: Optional[List[Dict[str, Any]]]) -> List[ToolCall]:
        """
        Parse tool calls from Ollama response.
        
        Args:
            tool_calls: List of tool call dicts from Ollama.
        
        Returns:
            List of ToolCall objects.
        """
        if not tool_calls:
            return []
        
        parsed: List[ToolCall] = []
        for i, tc in enumerate(tool_calls):
            function_info = tc.get("function", {})
            parsed.append(ToolCall(
                id=str(i),  # Ollama doesn't provide IDs, so we generate them
                type="function_call",
                function_name=function_info.get("name", ""),
                arguments=function_info.get("arguments", {})
            ))
        
        return parsed
    
    def _validate_model_sync(self, client: Any, model_id: str, auto_pull: bool = True) -> None:
        """
        Validate that the requested model is available via Ollama.
        If not found and auto_pull is True, attempt to pull the model.
        
        Args:
            client: The Ollama client instance.
            model_id: The model identifier to validate.
            auto_pull: Whether to attempt pulling the model if not found.
        
        Raises:
            ModelNotFoundError: If the model is not available and cannot be pulled.
        """
        try:
            # List available models
            response = client.list()
            models = response.get("models", [])
            available_model_names = [m.get("name", "").split(":")[0] for m in models]
            
            # Also check full names with tags
            full_model_names = [m.get("name", "") for m in models]
            
            model_base = model_id.split(":")[0]
            
            if model_id in full_model_names or model_base in available_model_names:
                return  # Model found
            
            if auto_pull:
                # Attempt to pull the model
                try:
                    # Pull model - this will stream progress
                    for _ in client.pull(model_id, stream=True):
                        pass  # Just consume the stream
                    return  # Successfully pulled
                except Exception as pull_error:
                    # Pull failed, raise ModelNotFoundError with available models
                    raise ModelNotFoundError(
                        model_id=model_id,
                        provider="Ollama",
                        available_models=full_model_names
                    ) from pull_error
            else:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Ollama",
                    available_models=full_model_names
                )
        
        except ModelNotFoundError:
            raise
        except Exception as e:
            # Wrap other errors
            mapped_error = self._handle_error(e)
            raise mapped_error
    
    async def _validate_model_async(self, client: Any, model_id: str, auto_pull: bool = True) -> None:
        """
        Asynchronously validate that the requested model is available via Ollama.
        If not found and auto_pull is True, attempt to pull the model.
        
        Args:
            client: The async Ollama client instance.
            model_id: The model identifier to validate.
            auto_pull: Whether to attempt pulling the model if not found.
        
        Raises:
            ModelNotFoundError: If the model is not available and cannot be pulled.
        """
        try:
            # List available models
            response = await client.list()
            models = response.get("models", [])
            available_model_names = [m.get("name", "").split(":")[0] for m in models]
            full_model_names = [m.get("name", "") for m in models]
            
            model_base = model_id.split(":")[0]
            
            if model_id in full_model_names or model_base in available_model_names:
                return  # Model found
            
            if auto_pull:
                # Attempt to pull the model
                try:
                    async for _ in await client.pull(model_id, stream=True):
                        pass  # Just consume the stream
                    return  # Successfully pulled
                except Exception as pull_error:
                    raise ModelNotFoundError(
                        model_id=model_id,
                        provider="Ollama",
                        available_models=full_model_names
                    ) from pull_error
            else:
                raise ModelNotFoundError(
                    model_id=model_id,
                    provider="Ollama",
                    available_models=full_model_names
                )
        
        except ModelNotFoundError:
            raise
        except Exception as e:
            mapped_error = self._handle_error(e)
            raise mapped_error


class AsyncOllamaProvider(AsyncBaseProvider, _OllamaProviderMixin):
    """
    Asynchronous provider implementation for Ollama's LLM API.
    
    This class provides async methods to generate text completions
    and stream responses from Ollama models (local or cloud),
    integrating robust error handling and retry mechanisms.
    
    Example:
        ```python
        from seqoria_agent.providers import AsyncOllamaProvider
        
        provider = AsyncOllamaProvider(model_id="gemma3")
        response = await provider.async_generate([
            {"role": "user", "content": "Why is the sky blue?"}
        ])
        print(response.text)
        ```
    """
    
    def __init__(
        self,
        model_id: str,
        host: Optional[str] = "http://localhost:11434",
        api_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the asynchronous Ollama provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gemma3", "llama3.2"). Required.
            host: Ollama server URL. Defaults to OLLAMA_HOST env or http://localhost:11434.
            api_key: Optional API key for cloud Ollama (ollama.com). Uses OLLAMA_API_KEY env if not provided.
            **kwargs: Additional parameters for model generation and client configuration:
                Generation options:
                    - temperature (float): Sampling temperature.
                    - top_p (float): Nucleus sampling parameter.
                    - top_k (int): Limits next token selection to K most likely.
                    - min_p (float): Minimum probability threshold.
                    - seed (int): Random seed for reproducible outputs.
                    - stop (str|list): Stop sequences.
                    - num_ctx (int): Context length size.
                    - num_predict (int): Maximum tokens to generate.
                Top-level params:
                    - format (str|dict): Output format ("json" or JSON schema).
                    - keep_alive (str|int): Model keep-alive duration.
                    - think (bool|str): Enable thinking output.
                Client options:
                    - max_retries (int): Maximum retries for retryable errors (default: 3).
                    - auto_pull (bool): Auto-pull model if not found (default: True).
        
        Raises:
            InvalidRequestError: If model_id is not provided.
            ConnectionError: If the async client cannot be initialized.
        """
        try:
            from ollama import AsyncClient
        except ImportError:
            raise EnvironmentError(
                message="ollama package is not installed. Install with: pip install ollama",
                missing_vars=[]
            )
        
        (
            self.host,
            self.headers,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.auto_pull
        ) = self._init_common(model_id, host, api_key, **kwargs)
        
        try:
            self.async_client = AsyncClient(host=self.host, headers=self.headers)
        except Exception as e:
            raise ConnectionError(provider="Ollama", message=str(e)) from e
    
    async def validate_model(self) -> None:
        """
        Validate that the configured model is available.
        
        If auto_pull is enabled and the model is not found, this will
        attempt to pull the model from the Ollama registry.
        
        Raises:
            ModelNotFoundError: If model is not available and cannot be pulled.
        """
        await self._validate_model_async(self.async_client, self.model_id, self.auto_pull)
    
    async def async_generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[LLMResponse, ToolCallList]:
        """
        Asynchronously generate a completion for the provided chat history.
        
        Args:
            chat_history: List of message dictionaries. Supports multimodal content:
                - {"role": "user", "content": "Hello"}
                - {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "base64": "iVBORw0KGgo..."}
                  ]}
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration (not fully supported by Ollama).
            model: Override the default model for this request.
            **kwargs: Generation parameters overriding defaults.
        
        Returns:
            LLMResponse with text content, or ToolCallList if tools were called.
        
        Raises:
            Various LLMError subclasses depending on the API response.
        """
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        
        messages = await self._prepare_chat_history_async(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)
        
        async def _call():
            response = await self.async_client.chat(
                model=effective_model,
                messages=messages,
                tools=ollama_tools,
                stream=False,
                options=options if options else None,
                **top_level
            )
            return response
        
        try:
            response = await self._execute_with_retry(_call)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise
        
        # Parse response
        message = response.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking")  # Thinking output when think=True
        tool_calls = message.get("tool_calls")
        
        # Build usage info
        usage = None
        prompt_eval_count = response.get("prompt_eval_count")
        eval_count = response.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            usage = LLMUsage(
                prompt_tokens=prompt_eval_count or 0,
                completion_tokens=eval_count or 0,
                total_tokens=(prompt_eval_count or 0) + (eval_count or 0)
            )
        
        # Return tool calls if present
        if tool_calls:
            parsed_tools = self._parse_tool_calls(tool_calls)
            return ToolCallList(
                model_id=response.get("model", effective_model),
                tool_calls=parsed_tools,
                usage=usage
            )
        
        return LLMResponse(
            model_id=response.get("model", effective_model),
            type="llm_response",
            text=content,
            thinking=thinking,
            usage=usage,
            stop_reason=response.get("done_reason")
        )
    
    async def async_stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """
        Asynchronously stream a completion for the provided chat history.
        
        Yields LLMResponse chunks as they arrive, with the final chunk
        containing usage information.
        
        Args:
            chat_history: List of message dictionaries (supports images).
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice configuration.
            model: Override the default model.
            **kwargs: Generation parameters.
        
        Yields:
            LLMResponse: Chunks with partial content.
            Final LLMResponse or ToolCallList with complete content and usage.
        """
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        
        messages = await self._prepare_chat_history_async(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)
        
        full_content = ""
        full_thinking = ""
        final_model = effective_model
        final_usage = None
        stop_reason = None
        tool_calls_accumulated: List[Dict[str, Any]] = []
        
        try:
            stream = await self.async_client.chat(
                model=effective_model,
                messages=messages,
                tools=ollama_tools,
                stream=True,
                options=options if options else None,
                **top_level
            )
            
            async for chunk in stream:
                message = chunk.get("message", {})
                content_delta = message.get("content", "")
                thinking_delta = message.get("thinking", "")
                
                # Accumulate tool calls
                if message.get("tool_calls"):
                    tool_calls_accumulated.extend(message["tool_calls"])
                
                # Accumulate thinking
                if thinking_delta:
                    full_thinking += thinking_delta
                    yield LLMResponse(
                        model_id=chunk.get("model", effective_model),
                        type="llm_delta",
                        text=None,
                        thinking=thinking_delta,
                        usage=None,
                        stop_reason=None
                    )
                
                if content_delta:
                    full_content += content_delta
                    yield LLMResponse(
                        model_id=chunk.get("model", effective_model),
                        type="llm_delta",
                        text=content_delta,
                        thinking=None,
                        usage=None,
                        stop_reason=None
                    )
                
                # Check for completion
                if chunk.get("done"):
                    final_model = chunk.get("model", effective_model)
                    stop_reason = chunk.get("done_reason")
                    
                    prompt_eval_count = chunk.get("prompt_eval_count")
                    eval_count = chunk.get("eval_count")
                    if prompt_eval_count is not None or eval_count is not None:
                        final_usage = LLMUsage(
                            prompt_tokens=prompt_eval_count or 0,
                            completion_tokens=eval_count or 0,
                            total_tokens=(prompt_eval_count or 0) + (eval_count or 0)
                        )
        
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise
        
        # Yield final response
        if tool_calls_accumulated:
            parsed_tools = self._parse_tool_calls(tool_calls_accumulated)
            yield ToolCallList(
                model_id=final_model,
                tool_calls=parsed_tools,
                usage=final_usage
            )
        else:
            yield LLMResponse(
                model_id=final_model,
                type="llm_response",
                text=full_content,
                thinking=full_thinking if full_thinking else None,
                usage=final_usage,
                stop_reason=stop_reason
            )


class OllamaProvider(BaseProvider, _OllamaProviderMixin):
    """
    Synchronous provider implementation for Ollama's LLM API.
    
    This class provides sync methods to generate text completions
    and stream responses from Ollama models (local or cloud).
    
    Example:
        ```python
        from seqoria_agent.providers import OllamaProvider
        
        provider = OllamaProvider(model_id="gemma3")
        response = provider.generate([
            {"role": "user", "content": "Why is the sky blue?"}
        ])
        print(response.text)
        ```
    """
    
    def __init__(
        self,
        model_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        validate_on_init: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the synchronous Ollama provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gemma3", "llama3.2"). Required.
            host: Ollama server URL. Defaults to OLLAMA_HOST env or http://localhost:11434.
            api_key: Optional API key for cloud Ollama (ollama.com). Uses OLLAMA_API_KEY env if not provided.
            validate_on_init: If True, validate (and potentially pull) the model on initialization.
            **kwargs: Additional parameters for model generation and client configuration:
                Generation options:
                    - temperature (float): Sampling temperature.
                    - top_p (float): Nucleus sampling parameter.
                    - top_k (int): Limits next token selection to K most likely.
                    - min_p (float): Minimum probability threshold.
                    - seed (int): Random seed for reproducible outputs.
                    - stop (str|list): Stop sequences.
                    - num_ctx (int): Context length size.
                    - num_predict (int): Maximum tokens to generate.
                Top-level params:
                    - format (str|dict): Output format ("json" or JSON schema).
                    - keep_alive (str|int): Model keep-alive duration.
                    - think (bool|str): Enable thinking output.
                Client options:
                    - max_retries (int): Maximum retries for retryable errors (default: 3).
                    - auto_pull (bool): Auto-pull model if not found (default: True).
        
        Raises:
            InvalidRequestError: If model_id is not provided.
            ConnectionError: If the client cannot be initialized.
            ModelNotFoundError: If validate_on_init is True and model cannot be found/pulled.
        """
        try:
            from ollama import Client
        except ImportError:
            raise EnvironmentError(
                message="ollama package is not installed. Install with: pip install ollama",
                missing_vars=[]
            )
        
        (
            self.host,
            self.headers,
            self.model_id,
            self.default_params,
            self.max_retries,
            self.auto_pull
        ) = self._init_common(model_id, host, api_key, **kwargs)
        
        try:
            self.client = Client(host=self.host, headers=self.headers)
        except Exception as e:
            raise ConnectionError(provider="Ollama", message=str(e)) from e
        
        if validate_on_init:
            self.validate_model()
    
    def validate_model(self) -> None:
        """
        Validate that the configured model is available.
        
        If auto_pull is enabled and the model is not found, this will
        attempt to pull the model from the Ollama registry.
        
        Raises:
            ModelNotFoundError: If model is not available and cannot be pulled.
        """
        self._validate_model_sync(self.client, self.model_id, self.auto_pull)
    
    def generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[LLMResponse, ToolCallList]:
        """
        Synchronously generate a completion for the provided chat history.
        
        Args:
            chat_history: List of message dictionaries. Supports multimodal content:
                - {"role": "user", "content": "Hello"}
                - {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "base64": "iVBORw0KGgo..."}
                  ]}
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration (not fully supported by Ollama).
            model: Override the default model for this request.
            **kwargs: Generation parameters overriding defaults.
        
        Returns:
            LLMResponse with text content, or ToolCallList if tools were called.
        
        Raises:
            Various LLMError subclasses depending on the API response.
        """
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        
        messages = self._prepare_chat_history_sync(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)
        
        def _call():
            response = self.client.chat(
                model=effective_model,
                messages=messages,
                tools=ollama_tools,
                stream=False,
                options=options if options else None,
                **top_level
            )
            return response
        
        try:
            response = self._execute_with_retry(_call)
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise
        
        # Parse response
        message = response.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking")  # Thinking output when think=True
        tool_calls = message.get("tool_calls")
        
        # Build usage info
        usage = None
        prompt_eval_count = response.get("prompt_eval_count")
        eval_count = response.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            usage = LLMUsage(
                prompt_tokens=prompt_eval_count or 0,
                completion_tokens=eval_count or 0,
                total_tokens=(prompt_eval_count or 0) + (eval_count or 0)
            )
        
        # Return tool calls if present
        if tool_calls:
            parsed_tools = self._parse_tool_calls(tool_calls)
            return ToolCallList(
                model_id=response.get("model", effective_model),
                tool_calls=parsed_tools,
                usage=usage
            )
        
        return LLMResponse(
            model_id=response.get("model", effective_model),
            type="llm_response",
            text=content,
            thinking=thinking,
            usage=usage,
            stop_reason=response.get("done_reason")
        )
    
    def stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        """
        Synchronously stream a completion for the provided chat history.
        
        Yields LLMResponse chunks as they arrive, with the final chunk
        containing usage information.
        
        Args:
            chat_history: List of message dictionaries (supports images).
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice configuration.
            model: Override the default model.
            **kwargs: Generation parameters.
        
        Yields:
            LLMResponse: Chunks with partial content.
            Final LLMResponse or ToolCallList with complete content and usage.
        """
        effective_model = model or self.model_id
        params = self._prepare_params(**kwargs)
        top_level, options = self._build_options(params)
        
        messages = self._prepare_chat_history_sync(chat_history)
        ollama_tools = self._convert_tools_for_ollama(tools)
        
        full_content = ""
        full_thinking = ""
        final_model = effective_model
        final_usage = None
        stop_reason = None
        tool_calls_accumulated: List[Dict[str, Any]] = []
        
        try:
            stream = self.client.chat(
                model=effective_model,
                messages=messages,
                tools=ollama_tools,
                stream=True,
                options=options if options else None,
                **top_level
            )
            
            for chunk in stream:
                message = chunk.get("message", {})
                content_delta = message.get("content", "")
                thinking_delta = message.get("thinking", "")
                
                # Accumulate tool calls
                if message.get("tool_calls"):
                    tool_calls_accumulated.extend(message["tool_calls"])
                
                # Accumulate thinking
                if thinking_delta:
                    full_thinking += thinking_delta
                    yield LLMResponse(
                        model_id=chunk.get("model", effective_model),
                        type="llm_delta",
                        text=None,
                        thinking=thinking_delta,
                        usage=None,
                        stop_reason=None
                    )
                
                if content_delta:
                    full_content += content_delta
                    yield LLMResponse(
                        model_id=chunk.get("model", effective_model),
                        type="llm_delta",
                        text=content_delta,
                        thinking=None,
                        usage=None,
                        stop_reason=None
                    )
                
                # Check for completion
                if chunk.get("done"):
                    final_model = chunk.get("model", effective_model)
                    stop_reason = chunk.get("done_reason")
                    
                    prompt_eval_count = chunk.get("prompt_eval_count")
                    eval_count = chunk.get("eval_count")
                    if prompt_eval_count is not None or eval_count is not None:
                        final_usage = LLMUsage(
                            prompt_tokens=prompt_eval_count or 0,
                            completion_tokens=eval_count or 0,
                            total_tokens=(prompt_eval_count or 0) + (eval_count or 0)
                        )
        
        except Exception as e:
            if not isinstance(e, LLMError):
                raise self._handle_error(e)
            raise
        
        # Yield final response
        if tool_calls_accumulated:
            parsed_tools = self._parse_tool_calls(tool_calls_accumulated)
            yield ToolCallList(
                model_id=final_model,
                tool_calls=parsed_tools,
                usage=final_usage
            )
        else:
            yield LLMResponse(
                model_id=final_model,
                type="llm_response",
                text=full_content,
                thinking=full_thinking if full_thinking else None,
                usage=final_usage,
                stop_reason=stop_reason
            )

