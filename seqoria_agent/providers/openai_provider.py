import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from openai import (AsyncOpenAI,
                    OpenAI,
                    APIConnectionError,
                    APIStatusError,
                    AuthenticationError,
                    BadRequestError,
                    ConflictError,
                    InternalServerError,
                    NotFoundError,
                    PermissionDeniedError,
                    RateLimitError,
                    UnprocessableEntityError)

from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider
from seqoria_agent.providers.errors import (ConnectionError,
                                            ContentFilterError,
                                            ContextLengthExceededError,
                                            EnvironmentError,
                                            InvalidRequestError,
                                            LLMError,
                                            ModelNotFoundError,
                                            QuotaExceededError,
                                            ServerError)

from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList
from seqoria_agent.models.chunk import AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult

# Use optimized JSON utilities
from seqoria_agent.client.ai_client.utils import json_dumps, json_loads

_EMPTY_JSON: str = "{}"


class _OpenAIProviderMixin:
    """
    Shared functionality between sync and async OpenAI providers.
    
    This mixin contains common methods that don't depend on sync/async behavior.
    Uses the OpenAI Responses API (v1/responses) for enhanced capabilities including
    reasoning/thinking token support.
    """
    
    def _init_common(self, model_id: str, api_key: Optional[str], base_url: Optional[str], **kwargs) -> tuple:
        """
        Common initialization logic shared between sync and async providers.
        
        Returns:
            tuple: (api_key, client_kwargs, model_id, default_params, max_retries)
        """
        # Prefer the explicitly supplied API key, otherwise fall back to the environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(message="OpenAI API key not provided", missing_vars=["OPENAI_API_KEY"])
        
        # Store max_retries for use in retry logic
        max_retries = kwargs.get('max_retries', 3)
        
        # Client configuration
        client_kwargs = {
            'timeout': kwargs.get('timeout', 30.0),
            'max_retries': 0,  # Disable automatic retries in the client; handled manually
        }

        # Optional custom base URL (Azure proxy, OpenAI-compatible gateway …)
        effective_base_url: Optional[str] = base_url or kwargs.get('base_url') or "https://api.openai.com/v1"
        if effective_base_url:
            client_kwargs['base_url'] = effective_base_url
        
        # model_id is required
        if not model_id:
            raise InvalidRequestError(message="model_id is required for OpenAI provider")
        
        effective_model_id = model_id

        # Initialize empty default_params - no hardcoded defaults
        default_params = {}
        
        # Define all valid parameter keys that can be passed to the OpenAI Responses API
        valid_query_keys = {
            # Generation parameters
            "max_output_tokens", "temperature", "top_p", "top_logprobs",
            # Reasoning configuration (for o-series and gpt-5 models)
            "reasoning",
            # Tool configuration
            "parallel_tool_calls", "max_tool_calls",
            # Response configuration
            "text", "truncation", "store",
            # Service configuration
            "service_tier", "metadata",
            # Caching
            "prompt_cache_key", "prompt_cache_retention",
            # Safety
            "safety_identifier",
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
            Dictionary of parameters ready for the OpenAI Responses API request.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def _handle_error(self, error: Exception) -> Exception:
        """
        Map OpenAI errors to our standard error types using isinstance checks.
        
        Args:
            error: Original error from OpenAI client.
            
        Returns
        -------
            Mapped error instance inheriting from LLMError.
        """
        # Check specific OpenAI error types first
        if isinstance(error, AuthenticationError):
            return EnvironmentError(message=f"OpenAI authentication failed: {error}", missing_vars=["OPENAI_API_KEY"])
        elif isinstance(error, RateLimitError):
            return QuotaExceededError(provider="OpenAI", limit_type="rate")
        elif isinstance(error, BadRequestError):
            error_message = str(object=error)
            if "context_length_exceeded" in error_message:
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
            elif "content_policy_violation" in error_message:
                 return ContentFilterError(provider="OpenAI", filter_reason=error_message)
            else:
                 return InvalidRequestError(message=f"Invalid request to OpenAI: {error}")
        elif isinstance(error, NotFoundError):
             return ModelNotFoundError(model_id=self.model_id, provider="OpenAI")
        elif isinstance(error, APIConnectionError):
             return ConnectionError(provider="OpenAI", message=f"Connection error: {error}")
        elif isinstance(error, InternalServerError):
             return ServerError(provider="OpenAI", status_code=error.status_code)
        elif isinstance(error, PermissionDeniedError):
             return InvalidRequestError(message=f"OpenAI permission denied: {error}")
        elif isinstance(error, UnprocessableEntityError):
             return ContentFilterError(provider="OpenAI", filter_reason=f"Unprocessable entity: {error}")
        elif isinstance(error, ConflictError):
             return ConnectionError(provider="OpenAI", message=f"Conflict error: {error}") # Treat as potentially retryable
        elif isinstance(error, APIStatusError):
            if 500 <= error.status_code < 600:
                return ServerError(provider="OpenAI", status_code=error.status_code)
            elif error.status_code == 429:
                return QuotaExceededError(provider="OpenAI", limit_type="rate")
            elif error.status_code == 400:
                 return InvalidRequestError(message=f"API Error {error.status_code}: {error}")
            else:
                 return LLMError(message=f"OpenAI API Error {error.status_code}: {error}")
        elif isinstance(error, asyncio.TimeoutError) or "timed out" in str(object=error).lower():
            return ConnectionError(provider="OpenAI", message="Request timed out")
            
        # Fallback for unexpected errors
        return LLMError(message=f"An unexpected OpenAI error occurred: {error}")
    
    def _convert_image_content_responses(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert image content blocks to OpenAI Responses API format.
        
        Supported input formats:
            - {"type": "text", "text": "..."} - Text content
            - {"type": "image", "url": "https://..."} - Image from URL
            - {"type": "image", "base64": "...", "media_type": "image/jpeg"} - Base64 encoded image
            - {"type": "image_url", ...} - Legacy OpenAI format
        
        Returns:
            List of content blocks in OpenAI Responses API format.
        """
        converted: List[Dict[str, Any]] = []
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image":
                # Convert our common format to Responses API input_image format
                if "url" in block:
                    converted.append({
                        "type": "input_image",
                        "image_url": block["url"],
                        "detail": block.get("detail", "auto")
                    })
                elif "base64" in block:
                    media_type = block.get("media_type", "image/jpeg")
                    data_url = f"data:{media_type};base64,{block['base64']}"
                    converted.append({
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": block.get("detail", "auto")
                    })
            elif block_type == "image_url":
                # Legacy OpenAI format - convert to Responses API format
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                converted.append({
                    "type": "input_image",
                    "image_url": url,
                    "detail": image_url.get("detail", "auto") if isinstance(image_url, dict) else "auto"
                })
            elif block_type == "input_text":
                # Already in Responses API format
                converted.append(block)
            elif block_type == "input_image":
                # Already in Responses API format
                converted.append(block)
            else:
                # Unknown type, try to convert as text
                if "text" in block:
                    converted.append({"type": "input_text", "text": block["text"]})
                else:
                    converted.append(block)
        
        return converted
    
    def _prepare_input_responses(
        self, 
        chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Prepare input for the OpenAI Responses API.
        
        Converts chat history into the Responses API input format.
        Returns a tuple of (instructions, input_items) where:
            - instructions: System/developer message to use as instructions
            - input_items: List of input items for the conversation
        
        The Responses API uses a different structure than Chat Completions:
            - System messages become 'instructions' parameter
            - User/assistant messages become input items with role
            - Tool calls become function_call items
            - Tool results become function_call_output items
        
        Supports vision/image content in messages.
        """
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for raw_msg in chat_history:
            # Handle pydantic model inputs first
            if isinstance(raw_msg, ToolCallRequest):
                # Tool call request -> function_call item
                input_items.append({
                    "type": "function_call",
                    "id": str(raw_msg.id),
                    "call_id": str(raw_msg.id),
                    "name": raw_msg.name,
                    "arguments": json_dumps(raw_msg.arguments) if raw_msg.arguments else _EMPTY_JSON
                })
                continue

            if isinstance(raw_msg, ToolCallResult):
                # Tool result -> function_call_output item
                input_items.append({
                    "type": "function_call_output",
                    "call_id": str(raw_msg.tool_call_id),
                    "output": str(raw_msg.result)
                })
                continue

            if isinstance(raw_msg, SystemMessage):
                # System message becomes instructions
                instructions_parts.append(raw_msg.content)
                continue
                
            if isinstance(raw_msg, (AssistantMessage, UserMessage)):
                content = raw_msg.content
                role = raw_msg.role
                
                # Convert role for Responses API (assistant stays assistant, user stays user)
                api_role = "assistant" if role == "assistant" else "user"
                
                # Check if content is multimodal (list of content blocks)
                if isinstance(content, list):
                    content = self._convert_image_content_responses(content)
                    input_items.append({
                        "type": "message",
                        "role": api_role,
                        "content": content
                    })
                else:
                    input_items.append({
                        "type": "message",
                        "role": api_role,
                        "content": content
                    })
                continue

            # Handle dict messages
            role = raw_msg.get("role")
            
            if role == "system" or role == "developer":
                # System/developer message becomes instructions
                content = raw_msg.get("content", "")
                instructions_parts.append(content)
                    
            elif role == "tool_call":
                # Tool call request
                input_items.append({
                    "type": "function_call",
                    "id": str(raw_msg.get("id")),
                    "call_id": str(raw_msg.get("id")),
                    "name": raw_msg.get("name"),
                    "arguments": json_dumps(raw_msg.get("arguments", {})) if raw_msg.get("arguments") else _EMPTY_JSON
                })
                
            elif role == "tool_result" or role == "tool":
                # Tool result
                input_items.append({
                    "type": "function_call_output",
                    "call_id": str(raw_msg.get("tool_call_id")),
                    "output": str(raw_msg.get("result", raw_msg.get("content", "")))
                })
                
            elif role in {"assistant", "user"}:
                content = raw_msg.get("content")
                
                # Check if content is multimodal (list of content blocks)
                if isinstance(content, list):
                    content = self._convert_image_content_responses(content)
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": content
                    })
                else:
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": content
                    })

        instructions: Optional[str] = "\n".join(instructions_parts) if instructions_parts else None
        return instructions, input_items
    
    def _format_tools_responses(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for the OpenAI Responses API.
        
        Converts from OpenAI Chat Completions format to Responses API format:
            Input: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            Output: [{"type": "function", "name": "...", "description": "...", "parameters": {...}, "strict": true}]
        """
        formatted_tools: List[Dict[str, Any]] = []
        
        for tool in tools:
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
                func_def = tool.get("function", tool)
                formatted_tools.append({
                    "type": "function",
                    "name": func_def.get("name"),
                    "description": func_def.get("description", ""),
                    "parameters": func_def.get("parameters", {}),
                    "strict": func_def.get("strict", True)
                })
            else:
                # Pass through other tool types (web_search, file_search, etc.)
                formatted_tools.append(tool)
        
        return formatted_tools
    
    def _parse_tool_call_sync(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        """
        Synchronous version of tool call parsing.
        """
        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            args = tc.get("arguments", {})
            try:
                args = json_loads(args) if isinstance(args, str) else args
            except Exception:
                args = {}

            processed_tool_calls.append(ToolCall(
                id=tc.get("id", tc.get("call_id", "")),
                type="function_call",
                function_name=tc.get("function_name", tc.get("name", "")),
                arguments=args
            )) 

        return processed_tool_calls
    
    def _extract_response_content(self, response: Any) -> tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Extract content from an OpenAI Responses API response.
        
        Returns:
            tuple: (text_content, thinking_content, tool_calls, stop_reason)
        """
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = None
        
        # Get status as stop reason
        if hasattr(response, 'status'):
            stop_reason = response.status
        
        # Process output items
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                item_type = getattr(output_item, 'type', None)
                
                if item_type == 'message':
                    # Extract text from message content
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_block in output_item.content:
                            content_type = getattr(content_block, 'type', None)
                            if content_type == 'output_text':
                                text_value = getattr(content_block, 'text', None)
                                if text_value:
                                    text_parts.append(text_value)
                                        
                elif item_type == 'reasoning':
                    # Extract reasoning/thinking content (o-series and gpt-5 models)
                    if hasattr(output_item, 'summary') and output_item.summary:
                        for summary_block in output_item.summary:
                            if hasattr(summary_block, 'text'):
                                thinking_value = summary_block.text
                                if thinking_value:
                                    thinking_parts.append(thinking_value)
                                        
                elif item_type == 'function_call':
                    # Extract function/tool call
                    call_id = getattr(output_item, 'call_id', getattr(output_item, 'id', ''))
                    name = getattr(output_item, 'name', '')
                    arguments = getattr(output_item, 'arguments', '{}')
                    
                    tool_calls.append({
                        "id": call_id,
                        "call_id": call_id,
                        "name": name,
                        "function_name": name,
                        "arguments": arguments
                    })
        
        # Try to use output_text convenience property if available
        text_content: Optional[str] = "".join(text_parts) if text_parts else None
        thinking_content: Optional[str] = "".join(thinking_parts) if thinking_parts else None

        if text_content is None and hasattr(response, 'output_text') and response.output_text:
            text_content = response.output_text
            
        return text_content, thinking_content, tool_calls, stop_reason
    
    def _extract_usage(self, response: Any) -> LLMUsage:
        """
        Extract usage information from an OpenAI Responses API response.
        """
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            usage_data["prompt_tokens"] = getattr(usage, 'input_tokens', 0) or 0
            usage_data["completion_tokens"] = getattr(usage, 'output_tokens', 0) or 0
            usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
            
        return LLMUsage(**usage_data)
    
    def _validate_model_sync(self, client: "OpenAI", model_id: str) -> None:
        """
        Validate that the requested model is available via the OpenAI API.
        
        Args:
            client: The OpenAI client instance.
            model_id: The model identifier to validate.
            
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        try:
            models_response = client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if model_id not in available_models:
                raise ModelNotFoundError(model_id=model_id,
                                         provider="OpenAI",
                                         available_models=available_models)
        except ModelNotFoundError:
            raise
        except Exception as e:
            # If we can't fetch models, wrap the error
            mapped_error = self._handle_error(error=e)
            raise mapped_error
    
    async def _validate_model_async(self, async_client: "AsyncOpenAI", model_id: str) -> None:
        """
        Asynchronously validate that the requested model is available via the OpenAI API.
        
        Args:
            async_client: The async OpenAI client instance.
            model_id: The model identifier to validate.
            
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        try:
            models_response = await async_client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if model_id not in available_models:
                raise ModelNotFoundError(model_id=model_id,
                                         provider="OpenAI",
                                         available_models=available_models)
        except ModelNotFoundError:
            raise
        except Exception as e:
            # If we can't fetch models, wrap the error
            mapped_error = self._handle_error(error=e)
            raise mapped_error


class AsyncOpenAIProvider(AsyncBaseProvider, _OpenAIProviderMixin):
    """
    Asynchronous provider implementation for OpenAI's LLM API.
    
    This class provides async methods to generate text completions
    and stream responses from OpenAI's large language models using
    the Responses API, integrating robust error handling and retry mechanisms.
    
    The Responses API provides enhanced capabilities including:
        - Reasoning/thinking token support for o-series and gpt-5 models
        - Built-in tools (web search, file search, code interpreter)
        - Conversation management
        - Enhanced streaming
    """
    
    def __init__(self, model_id: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize the asynchronous OpenAI provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gpt-4o", "o3", "gpt-5"). Required.
            api_key: Explicit API key for authentication. If None, falls back to environment variable.
            base_url: Custom OpenAI-compatible endpoint URL.
            **kwargs: Additional parameters for model generation and client configuration:
                Query kwargs (passed to OpenAI Responses API):
                    - temperature (float): Sampling temperature (0-2)
                    - max_output_tokens (int): Maximum tokens to generate
                    - top_p (float): Nucleus sampling parameter
                    - reasoning (dict): Reasoning configuration for o-series/gpt-5 models
                        e.g., {"effort": "high"} or {"effort": "medium", "summary": "auto"}
                    - truncation (str): "auto" or "disabled"
                    - store (bool): Whether to store the response (default: True)
                    - service_tier (str): "auto", "default", "flex", or "priority"
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
        
        Raises
        ------
            EnvironmentError: If OPENAI_API_KEY environment variable is not set.
            ConnectionError: If the async client cannot be initialized.
        """
        api_key, client_kwargs, self.model_id, self.default_params, self.max_retries = self._init_common(
            model_id, api_key, base_url, **kwargs
        )
        
        try:
            # Create and assign the async client
            self.async_client = AsyncOpenAI(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="OpenAI") from e
    
    async def validate_model(self) -> None:
        """
        Validate that the configured model is available via the OpenAI API.
        
        Call this method after initialization to verify the model exists.
        
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        
        Example:
            provider = AsyncOpenAIProvider(model_id="gpt-4o", api_key="...")
            await provider.validate_model()  # Raises if model not found
        """
        await self._validate_model_async(async_client=self.async_client, model_id=self.model_id)
    
    # Reuse shared implementation from AsyncBaseProvider
    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)
    
    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:
        """
        Parse a tool call into a ToolCall object.
        """
        return self._parse_tool_call_sync(tool_call)
    
    async def async_generate(self,
                             chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                             tools: Optional[List[Dict[str, Any]]] = None,
                             tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                             model: Optional[str] = None,
                             **kwargs) -> Union[LLMResponse, ToolCallList]:
        """
        Asynchronously generate a completion for the provided chat history.
        
        Uses the OpenAI Responses API which provides enhanced capabilities including
        reasoning/thinking token support for o-series and gpt-5 models.
        
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
            tools: List of tool dictionaries for tool-based generation:
                [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            tool_choice: How the model should select tools. Can be:
                - "auto": Let the model decide
                - "none": Don't use any tools
                - "required": Must use at least one tool
                - {"type": "function", "name": "..."}: Use a specific function
            model: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_output_tokens).
            
        Returns
        -------
            Union[LLMResponse, ToolCallList]: Response containing:
                - text (str | None): Generated text string
                - thinking (str | None): Reasoning/thinking content (for o-series/gpt-5 models)
                - usage (LLMUsage): Token usage information
                - stop_reason (str | None): Response status
            
        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        params = self._prepare_params(**kwargs)
        
        # Prepare input from chat history
        instructions, input_items = self._prepare_input_responses(chat_history)
        
        # Add instructions if present
        if instructions:
            params["instructions"] = instructions
        
        # Add tools if provided
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        
        async def _generate() -> Union[LLMResponse, ToolCallList]:
            response = await self.async_client.responses.create(
                model=effective_model,
                input=input_items,
                **params
            )
            
            # Extract content from response
            text_content, thinking_content, tool_calls, stop_reason = self._extract_response_content(response)
            usage_obj = self._extract_usage(response)
            
            if tool_calls:
                tool_calls_parsed = await self._parse_tool_call(tool_calls)
                return ToolCallList(model_id=effective_model,
                                    tool_calls=tool_calls_parsed,
                                    usage=usage_obj)

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=text_content,
                               thinking=thinking_content,
                               usage=usage_obj,
                               stop_reason=stop_reason)
        
        return await self._execute_with_retry(operation=_generate)
    
    async def async_stream(self, 
                           chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                           model: Optional[str] = None,
                           **kwargs) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """
        Asynchronously stream a completion for the provided chat history.
        
        Yields content chunks as they arrive, and finally yields a
        LLMResponse containing the full concatenated text and token usage.
        
        Uses the OpenAI Responses API streaming which provides enhanced capabilities
        including reasoning/thinking token support for o-series and gpt-5 models.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Args:
            chat_history: List of message dictionaries.
            tools: List of tool dictionaries for tool-based generation.
            tool_choice: How the model should select tools.
            model: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults.
            
        Yields
        ------
            LLMResponse: Delta chunks with text and/or thinking content.
            LLMResponse: Final response with full content, usage info, and stop_reason.
            ToolCallList: If the model made tool calls.

        Raises
        ------
            Same exceptions as async_generate() method, potentially during iteration.
        """
        params = self._prepare_params(**kwargs)
        
        # Prepare input from chat history
        instructions, input_items = self._prepare_input_responses(chat_history)
        
        # Add instructions if present
        if instructions:
            params["instructions"] = instructions
        
        # Add tools if provided
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        
        async def _create_stream():
            return await self.async_client.responses.create(
                model=effective_model,
                input=input_items,
                stream=True,
                **params
            )
        
        completion_parts: List[str] = []
        thinking_parts: List[str] = []

        # Tool-call tracking (avoid O(n^2) scans during streaming)
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
        final_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stop_reason = None

        try:
            async for event in await self._execute_with_retry(operation=_create_stream):
                event_type = getattr(event, 'type', None)
                
                # Handle text delta events
                if event_type == 'response.output_text.delta':
                    delta_text = getattr(event, 'delta', '')
                    if delta_text:
                        completion_parts.append(delta_text)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=delta_text,
                                          thinking=None,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle reasoning/thinking delta events
                elif event_type == 'response.reasoning.delta':
                    delta_thinking = getattr(event, 'delta', '')
                    if delta_thinking:
                        thinking_parts.append(delta_thinking)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=None,
                                          thinking=delta_thinking,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle reasoning summary delta
                elif event_type == 'response.reasoning_summary_text.delta':
                    delta_thinking = getattr(event, 'delta', '')
                    if delta_thinking:
                        thinking_parts.append(delta_thinking)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=None,
                                          thinking=delta_thinking,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle function call start
                elif event_type == 'response.function_call_arguments.start':
                    call_id = getattr(event, 'call_id', getattr(event, 'id', ''))
                    name = getattr(event, 'name', '')
                    current_tool_call = {
                        "id": call_id,
                        "call_id": call_id,
                        "name": name,
                        "function_name": name,
                        "arguments": ""
                    }
                    current_tool_args_parts = []
                
                # Handle function call arguments delta
                elif event_type == 'response.function_call_arguments.delta':
                    if current_tool_call:
                        delta_args = getattr(event, 'delta', '')
                        if delta_args:
                            current_tool_args_parts.append(delta_args)
                
                # Handle function call done
                elif event_type == 'response.function_call_arguments.done':
                    if current_tool_call:
                        if current_tool_args_parts:
                            current_tool_call["arguments"] = "".join(current_tool_args_parts)
                        _track_tool_call(current_tool_call)
                        current_tool_call = None
                        current_tool_args_parts = []
                
                # Handle output item added (for function calls)
                elif event_type == 'response.output_item.added':
                    item = getattr(event, 'item', None)
                    if item and getattr(item, 'type', None) == 'function_call':
                        call_id = getattr(item, 'call_id', getattr(item, 'id', ''))
                        name = getattr(item, 'name', '')
                        initial_args = getattr(item, 'arguments', '') or ""
                        current_tool_call = {
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": initial_args
                        }
                        current_tool_args_parts = [initial_args] if initial_args else []
                
                # Handle output item done (for function calls)
                elif event_type == 'response.output_item.done':
                    item = getattr(event, 'item', None)
                    if item and getattr(item, 'type', None) == 'function_call':
                        call_id = getattr(item, 'call_id', getattr(item, 'id', ''))
                        name = getattr(item, 'name', '')
                        arguments = getattr(item, 'arguments', '{}')
                        _track_tool_call({
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": arguments,
                        })
                        current_tool_call = None
                        current_tool_args_parts = []
                
                # Handle response completed
                elif event_type == 'response.completed':
                    response = getattr(event, 'response', None)
                    if response:
                        stop_reason = getattr(response, 'status', None)
                        # Extract final usage
                        if hasattr(response, 'usage') and response.usage:
                            usage = response.usage
                            final_usage_data["prompt_tokens"] = getattr(usage, 'input_tokens', 0) or 0
                            final_usage_data["completion_tokens"] = getattr(usage, 'output_tokens', 0) or 0
                            final_usage_data["total_tokens"] = final_usage_data["prompt_tokens"] + final_usage_data["completion_tokens"]

            usage_obj = LLMUsage(**final_usage_data)

            full_completion_text = "".join(completion_parts)
            full_thinking_text = "".join(thinking_parts)
            tool_calls: List[Dict[str, Any]] = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id
            
            if tool_calls:
                tool_calls_obj = await self._parse_tool_call(tool_call=tool_calls)
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls_obj,
                                   usage=usage_obj)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text if full_completion_text else None,
                                  thinking=full_thinking_text if full_thinking_text else None,
                                  usage=usage_obj,
                                  stop_reason=stop_reason)

        except Exception as e:
            # Handle errors occurring during async stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error


class OpenAIProvider(BaseProvider, _OpenAIProviderMixin):
    """
    Synchronous provider implementation for OpenAI's LLM API.
    
    This class provides sync methods to generate text completions
    and stream responses from OpenAI's large language models using
    the Responses API, integrating robust error handling and retry mechanisms.
    
    The Responses API provides enhanced capabilities including:
        - Reasoning/thinking token support for o-series and gpt-5 models
        - Built-in tools (web search, file search, code interpreter)
        - Conversation management
        - Enhanced streaming
    """
    
    def __init__(self, model_id: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize the synchronous OpenAI provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gpt-4o", "o3", "gpt-5"). Required.
            api_key: Explicit API key for authentication. If None, falls back to environment variable.
            base_url: Custom OpenAI-compatible endpoint URL.
            **kwargs: Additional parameters for model generation and client configuration:
                Query kwargs (passed to OpenAI Responses API):
                    - temperature (float): Sampling temperature (0-2)
                    - max_output_tokens (int): Maximum tokens to generate
                    - top_p (float): Nucleus sampling parameter
                    - reasoning (dict): Reasoning configuration for o-series/gpt-5 models
                        e.g., {"effort": "high"} or {"effort": "medium", "summary": "auto"}
                    - truncation (str): "auto" or "disabled"
                    - store (bool): Whether to store the response (default: True)
                    - service_tier (str): "auto", "default", "flex", or "priority"
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
        
        Raises
        ------
            EnvironmentError: If OPENAI_API_KEY environment variable is not set.
            ConnectionError: If the sync client cannot be initialized.
        """
        api_key, client_kwargs, self.model_id, self.default_params, self.max_retries = self._init_common(
            model_id, api_key, base_url, **kwargs
        )
        
        try:
            # Create and assign the sync client
            self.client = OpenAI(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="OpenAI") from e
        
        # Validate that the requested model is available
        self._validate_model_sync(client=self.client, model_id=self.model_id)
    
    def validate_model(self) -> None:
        """
        Validate that the configured model is available via the OpenAI API.
        
        This is called automatically during initialization, but can be called
        again if needed (e.g., after changing model_id).
        
        Raises:
            ModelNotFoundError: If the model is not available, includes list of available models.
        """
        self._validate_model_sync(client=self.client, model_id=self.model_id)
    
    def generate(self,
                 chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                 model: Optional[str] = None,
                 **kwargs) -> Union[LLMResponse, ToolCallList]:
        """
        Synchronously generate a completion for the provided chat history.
        
        Uses the OpenAI Responses API which provides enhanced capabilities including
        reasoning/thinking token support for o-series and gpt-5 models.
        
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
            tools: List of tool dictionaries for tool-based generation.
            tool_choice: How the model should select tools.
            model: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_output_tokens).
            
        Returns
        -------
            Union[LLMResponse, ToolCallList]: Response containing:
                - text (str | None): Generated text string
                - thinking (str | None): Reasoning/thinking content (for o-series/gpt-5 models)
                - usage (LLMUsage): Token usage information
                - stop_reason (str | None): Response status
            
        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        params = self._prepare_params(**kwargs)
        
        # Prepare input from chat history
        instructions, input_items = self._prepare_input_responses(chat_history)
        
        # Add instructions if present
        if instructions:
            params["instructions"] = instructions
        
        # Add tools if provided
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        
        def _generate() -> Union[LLMResponse, ToolCallList]:
            response = self.client.responses.create(
                model=effective_model,
                input=input_items,
                **params
            )
            
            # Extract content from response
            text_content, thinking_content, tool_calls, stop_reason = self._extract_response_content(response)
            usage_obj = self._extract_usage(response)
            
            if tool_calls:
                tool_calls_parsed = self._parse_tool_call_sync(tool_calls)
                return ToolCallList(model_id=effective_model,
                                    tool_calls=tool_calls_parsed,
                                    usage=usage_obj)

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=text_content,
                               thinking=thinking_content,
                               usage=usage_obj,
                               stop_reason=stop_reason)
        
        return self._execute_with_retry(operation=_generate)
    
    def stream(self, 
               chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]], 
               tools: Optional[List[Dict[str, Any]]] = None,
               tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
               model: Optional[str] = None,
               **kwargs) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        """
        Synchronously stream a completion for the provided chat history.
        
        Yields content chunks as they arrive, and finally yields a
        LLMResponse containing the full concatenated text and token usage.
        
        Uses the OpenAI Responses API streaming which provides enhanced capabilities
        including reasoning/thinking token support for o-series and gpt-5 models.
        
        Supports vision/images by passing multimodal content:
            [{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]}]
        
        Args:
            chat_history: List of message dictionaries.
            tools: List of tool dictionaries for tool-based generation.
            tool_choice: How the model should select tools.
            model: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults.
            
        Yields
        ------
            LLMResponse: Delta chunks with text and/or thinking content.
            LLMResponse: Final response with full content, usage info, and stop_reason.
            ToolCallList: If the model made tool calls.

        Raises
        ------
            Same exceptions as generate() method, potentially during iteration.
        """
        params = self._prepare_params(**kwargs)
        
        # Prepare input from chat history
        instructions, input_items = self._prepare_input_responses(chat_history)
        
        # Add instructions if present
        if instructions:
            params["instructions"] = instructions
        
        # Add tools if provided
        if tools is not None:
            params["tools"] = self._format_tools_responses(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        effective_model = model or self.model_id
        
        def _create_stream():
            return self.client.responses.create(
                model=effective_model,
                input=input_items,
                stream=True,
                **params
            )
        
        completion_parts: List[str] = []
        thinking_parts: List[str] = []

        # Tool-call tracking (avoid O(n^2) scans during streaming)
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
        final_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stop_reason = None

        try:
            for event in self._execute_with_retry(operation=_create_stream):
                event_type = getattr(event, 'type', None)
                
                # Handle text delta events
                if event_type == 'response.output_text.delta':
                    delta_text = getattr(event, 'delta', '')
                    if delta_text:
                        completion_parts.append(delta_text)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=delta_text,
                                          thinking=None,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle reasoning/thinking delta events
                elif event_type == 'response.reasoning.delta':
                    delta_thinking = getattr(event, 'delta', '')
                    if delta_thinking:
                        thinking_parts.append(delta_thinking)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=None,
                                          thinking=delta_thinking,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle reasoning summary delta
                elif event_type == 'response.reasoning_summary_text.delta':
                    delta_thinking = getattr(event, 'delta', '')
                    if delta_thinking:
                        thinking_parts.append(delta_thinking)
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=None,
                                          thinking=delta_thinking,
                                          usage=None,
                                          stop_reason=None)
                
                # Handle function call start
                elif event_type == 'response.function_call_arguments.start':
                    call_id = getattr(event, 'call_id', getattr(event, 'id', ''))
                    name = getattr(event, 'name', '')
                    current_tool_call = {
                        "id": call_id,
                        "call_id": call_id,
                        "name": name,
                        "function_name": name,
                        "arguments": ""
                    }
                    current_tool_args_parts = []
                
                # Handle function call arguments delta
                elif event_type == 'response.function_call_arguments.delta':
                    if current_tool_call:
                        delta_args = getattr(event, 'delta', '')
                        if delta_args:
                            current_tool_args_parts.append(delta_args)
                
                # Handle function call done
                elif event_type == 'response.function_call_arguments.done':
                    if current_tool_call:
                        if current_tool_args_parts:
                            current_tool_call["arguments"] = "".join(current_tool_args_parts)
                        _track_tool_call(current_tool_call)
                        current_tool_call = None
                        current_tool_args_parts = []
                
                # Handle output item added (for function calls)
                elif event_type == 'response.output_item.added':
                    item = getattr(event, 'item', None)
                    if item and getattr(item, 'type', None) == 'function_call':
                        call_id = getattr(item, 'call_id', getattr(item, 'id', ''))
                        name = getattr(item, 'name', '')
                        initial_args = getattr(item, 'arguments', '') or ""
                        current_tool_call = {
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": initial_args
                        }
                        current_tool_args_parts = [initial_args] if initial_args else []
                
                # Handle output item done (for function calls)
                elif event_type == 'response.output_item.done':
                    item = getattr(event, 'item', None)
                    if item and getattr(item, 'type', None) == 'function_call':
                        call_id = getattr(item, 'call_id', getattr(item, 'id', ''))
                        name = getattr(item, 'name', '')
                        arguments = getattr(item, 'arguments', '{}')
                        _track_tool_call({
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "function_name": name,
                            "arguments": arguments,
                        })
                        current_tool_call = None
                        current_tool_args_parts = []
                
                # Handle response completed
                elif event_type == 'response.completed':
                    response = getattr(event, 'response', None)
                    if response:
                        stop_reason = getattr(response, 'status', None)
                        # Extract final usage
                        if hasattr(response, 'usage') and response.usage:
                            usage = response.usage
                            final_usage_data["prompt_tokens"] = getattr(usage, 'input_tokens', 0) or 0
                            final_usage_data["completion_tokens"] = getattr(usage, 'output_tokens', 0) or 0
                            final_usage_data["total_tokens"] = final_usage_data["prompt_tokens"] + final_usage_data["completion_tokens"]

            usage_obj = LLMUsage(**final_usage_data)

            full_completion_text = "".join(completion_parts)
            full_thinking_text = "".join(thinking_parts)
            tool_calls: List[Dict[str, Any]] = [tool_calls_by_id[cid] for cid in tool_call_order] + tool_calls_no_id
            
            if tool_calls:
                tool_calls_obj = self._parse_tool_call_sync(tool_call=tool_calls)
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls_obj,
                                   usage=usage_obj)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text if full_completion_text else None,
                                  thinking=full_thinking_text if full_thinking_text else None,
                                  usage=usage_obj,
                                  stop_reason=stop_reason)

        except Exception as e:
            # Handle errors occurring during stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error
