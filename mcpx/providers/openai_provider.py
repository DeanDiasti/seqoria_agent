import os
from dotenv import load_dotenv
from typing import Dict, Optional, Any, List, AsyncGenerator, Union
import openai
import asyncio
import random # Import random for jitter in backoff

# Import specific exceptions from the openai library for better error handling
from openai import (APIConnectionError, 
                    RateLimitError, 
                    APIStatusError, 
                    BadRequestError, 
                    AuthenticationError,
                    NotFoundError,
                    ConflictError,
                    InternalServerError,
                    PermissionDeniedError,
                    UnprocessableEntityError)
from openai.types.chat import ChatCompletionChunk

# Use a relative import since we're in the same directory
from .schemas import LLMResponse, ToolCall, LLMUsage
from .errors import (LLMError, 
                    EnvironmentError, 
                    ConnectionError, 
                    ModelNotFoundError, 
                    ContextLengthExceededError, 
                    ContentFilterError, 
                    MaxRetriesExceededError, 
                    InvalidRequestError, 
                    QuotaExceededError, 
                    ServerError)
from .base_provider import BaseProvider

load_dotenv()

class OpenAIProvider(BaseProvider):
    """
    Asynchronous provider implementation for OpenAI's LLM API.
    
    This class provides async methods to generate text completions
    and stream responses from OpenAI's large language models,
    integrating robust error handling and retry mechanisms.
    """
    
    # Default model to use if requested model is unavailable or none is specified
    DEFAULT_MODEL = "gpt-4o" 
    
    def __init__(self, model_id: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the asynchronous OpenAI provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gpt-4o", "gpt-3.5-turbo").
                      If None, falls back to DEFAULT_MODEL. Validation occurs during API calls.
            **kwargs: Additional parameters for model generation and client configuration:
                Query kwargs (passed to OpenAI API):
                    - temperature (float): Sampling temperature (default: 0.7)
                    - max_tokens (int): Maximum tokens to generate (default: 1024).
                    - top_p (float): Nucleus sampling parameter (default: 1.0)
                    - frequency_penalty (float): Penalty for frequency (default: 0.0)
                    - presence_penalty (float): Penalty for presence (default: 0.0)
                    - stop (Union[str, List[str]]): Sequence(s) that stop generation.
                    - seed (int): Seed for deterministic sampling.
                    - response_format (dict): e.g., {"type": "json_object"}
                    # Add other valid OpenAI parameters as needed
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
                    - base_url (str): Custom OpenAI-compatible endpoint URL.
        
        Raises:
            EnvironmentError: If OPENAI_API_KEY environment variable is not set.
            ConnectionError: If the async client cannot be initialized.
        """

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(message="OpenAI API key not found in environment variables",
                                   missing_vars=["OPENAI_API_KEY"])
        
        # Store max_retries for use in retry logic
        self.max_retries = kwargs.get('max_retries', 3)
        
        
        # --------------------------------------------------------------
        # Async OpenAI client configuration
        # --------------------------------------------------------------

        client_kwargs = {
            'timeout': kwargs.get('timeout', 30.0),
            'max_retries': 0,  # Disable automatic retries in the client; handled manually
        }

        # Optional custom base URL (Azure proxy, OpenAI-compatible gateway …)
        base_url: Optional[str] = kwargs.get('base_url') or "https://api.openai.com/v1"
        if base_url:
            client_kwargs['base_url'] = base_url
        
        try:
            # Create and assign the async client
            self.async_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="OpenAI") from e
        
        # Store the requested model_id or use the default if None
        self.model_id = model_id if model_id else self.DEFAULT_MODEL

        # Set default parameters using OpenAI's parameter names
        self.default_params = {"max_tokens": 1024,
                               "temperature": 0.7,
                               "top_p": 1.0,
                               "frequency_penalty": 0.0,
                               "presence_penalty": 0.0}
        
        # Update default_params with any valid query kwargs provided in __init__
        valid_query_keys = self.default_params.keys() | {"stop", "seed", "response_format", "user", "logit_bias", "logprobs", "top_logprobs"} # Add other valid keys
        for key, value in kwargs.items():
            if key in valid_query_keys:
                self.default_params[key] = value

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare request parameters by merging defaults with overrides.
        Removes parameters with None values.

        Args:
            **kwargs: Parameter overrides provided to async_generate/async_stream.
            
        Returns:
            Dictionary of parameters ready for the OpenAI API request.
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
            
        Returns:
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
                        if part == 'tokens': tokens = int(x=parts[i-1])
                        if part == 'limit': max_tokens = int(x=parts[i-1])
                 except (ValueError, IndexError): pass
                 return ContextLengthExceededError(model_id=self.model_id, tokens=tokens, max_tokens=max_tokens)
            elif "content_policy_violation" in error_message:
                 return ContentFilterError(provider="OpenAI", filter_reason=error_message)
            else:
                 return InvalidRequestError(message=f"Invalid request to OpenAI: {error}")
        elif isinstance(error, NotFoundError):
             return ModelNotFoundError(model_id=self.model_id, provider="OpenAI", fallback_model=self.DEFAULT_MODEL)
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
        elif isinstance(error, asyncio.TimeoutError) or "timed out" in str(object=error).lower(): # Catch asyncio timeouts
            return ConnectionError(provider="OpenAI", message="Request timed out")
            
        # Fallback for unexpected errors
        return LLMError(message=f"An unexpected OpenAI error occurred: {error}")
    
    # Reuse shared implementation from BaseProvider (introduced in v0.2)
    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation)
    
    async def async_generate(self, 
                             chat_history: List[Dict[str, Any]], 
                             tools: Optional[List[Dict[str, Any]]] = None,
                             tool_choice: Optional[Dict[str, Any]] = None,
                             **kwargs) -> LLMResponse:
        """
        Asynchronously generate a completion for the provided chat history.
        
        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: List of tool dictionaries for tool-based generation in OpenAI format:
                [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            tool_choice: Dictionary specifying the tool choice (e.g., "auto", "none", or {"type": "function", "function": {"name": "..."}}).
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).
            
        Returns:
            LLMResponse: Response containing:
                - text (str | None): Generated text string. None if generation fails or is empty.
                - usage (LLMUsage): Token usage information
                - tool_calls (list | None): List of tool calls in OpenAI format:
                    [{"id": "call_123", "type": "function", "function": {"name": "...", "arguments": "..."}}]
                - stop_reason (str | None): Reason why generation stopped
            
        Raises:
            Various LLMError subclasses depending on the API response or connection issues.
        """

        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        async def _generate() -> LLMResponse:
            completion = await self.async_client.chat.completions.create(model=self.model_id,
                                                                          messages=chat_history,
                                                                          stream=False,
                                                                          **params)
            
            content = None
            tool_calls = []
            
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content
                # Extract tool calls if present - format according to OpenAI specification
                if hasattr(completion.choices[0].message, 'tool_calls') and completion.choices[0].message.tool_calls:
                    for tool_call in completion.choices[0].message.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type,  # Should be "function"
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments  # Keep as JSON string
                            }
                        })

            usage_data = {"prompt_tokens": 0, 
                          "completion_tokens": 0, 
                          "total_tokens": 0}
            
            if completion.usage:
                usage_data = {"prompt_tokens": completion.usage.prompt_tokens or 0,
                              "completion_tokens": completion.usage.completion_tokens or 0,
                              "total_tokens": completion.usage.total_tokens or 0}

            # Extract stop reason
            stop_reason = None
            if completion.choices and completion.choices[0].finish_reason:
                stop_reason = completion.choices[0].finish_reason

            usage_obj = LLMUsage(**usage_data)
            tool_objs = [ToolCall(**tc) for tc in tool_calls] if tool_calls else None
            return LLMResponse(
                model_id=self.model_id,
                text=content,
                tool_calls=tool_objs,
                usage=usage_obj,
                stop_reason=stop_reason,
            )
        
        return await self._execute_with_retry(operation=_generate)
    
    async def async_stream(self, 
                           chat_history: List[Dict[str, Any]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Dict[str, Any]] = None,
                           **kwargs) -> AsyncGenerator[Union[str, LLMResponse], None]:
        """
        Asynchronously stream a completion for the provided chat history.
        
        Yields content chunks (str) as they arrive, and finally yields a
        LLMResponse containing the full concatenated text and token usage.
        
        Args:
            chat_history: List of message dictionaries.
            tools: List of tool dictionaries for tool-based generation in OpenAI format:
                [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            tool_choice: Dictionary specifying the tool choice (e.g., "auto", "none", or {"type": "function", "function": {"name": "..."}}).
            **kwargs: Generation parameters overriding defaults.
            
        Yields:
            str: Text chunks of the generated completion.
            LLMResponse: Final LLMResponse with "text" (full content), "usage" info, and "tool_calls".
                Example: LLMResponse(model_id="gpt-4o", text="Full response", tool_calls=[...], usage=LLMUsage(...), stop_reason="...")

        Raises:
            Same exceptions as async_generate() method, potentially during iteration.
        """

        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        async def _create_stream() -> openai.AsyncStream[ChatCompletionChunk]:
            # Use the async client here
            return await self.async_client.chat.completions.create(model=self.model_id,
                                                                   messages=chat_history,
                                                                   stream=True,
                                                                   stream_options={"include_usage": True},
                                                                   **params)
        
        full_completion_text = ""
        tool_calls = []
        final_usage_data = {"prompt_tokens": 0, 
                            "completion_tokens": 0, 
                            "total_tokens": 0}
        stop_reason = None

        try:
            # Asynchronously iterate through the stream chunks
            async for chunk in await self._execute_with_retry(operation=_create_stream):
                 # Check if it's the final chunk with usage data
                 if chunk.usage:
                     final_usage_data = {"prompt_tokens": chunk.usage.prompt_tokens or 0,
                                         "completion_tokens": chunk.usage.completion_tokens or 0,
                                         "total_tokens": chunk.usage.total_tokens or 0}

                 # Check if it's a content chunk
                 if chunk.choices and chunk.choices[0].delta:
                     delta = chunk.choices[0].delta
                     
                     # Handle text content
                     if delta.content:
                         full_completion_text += delta.content
                         yield delta.content
                     
                     # Handle tool calls - format according to OpenAI specification
                     if hasattr(delta, 'tool_calls') and delta.tool_calls:
                         for tool_call_delta in delta.tool_calls:
                             if tool_call_delta.index is not None:
                                 # Ensure we have enough slots in the tool_calls list
                                 while len(tool_calls) <= tool_call_delta.index:
                                     tool_calls.append({
                                         "id": "",
                                         "type": "function",
                                         "function": {
                                             "name": "",
                                             "arguments": ""
                                         }
                                     })
                                 
                                 # Update the tool call at the specified index
                                 current_tool_call = tool_calls[tool_call_delta.index]
                                 
                                 # Update ID if provided
                                 if tool_call_delta.id:
                                     current_tool_call["id"] = tool_call_delta.id
                                 
                                 # Update type if provided
                                 if tool_call_delta.type:
                                     current_tool_call["type"] = tool_call_delta.type
                                 
                                 # Update function details if provided
                                 if tool_call_delta.function:
                                     if tool_call_delta.function.name:
                                         current_tool_call["function"]["name"] = tool_call_delta.function.name
                                     if tool_call_delta.function.arguments:
                                         current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments
                     
                     # Extract finish reason
                     if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                         stop_reason = chunk.choices[0].finish_reason
            
            # After the loop finishes, yield the final LLMResponse
            usage_obj = LLMUsage(**final_usage_data)
            tool_objs = [ToolCall(**tc) for tc in tool_calls] if tool_calls else None
            yield LLMResponse(
                model_id=self.model_id,
                text=full_completion_text,
                tool_calls=tool_objs,
                usage=usage_obj,
                stop_reason=stop_reason,
            )

        except Exception as e:
            # Handle errors occurring during async stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error