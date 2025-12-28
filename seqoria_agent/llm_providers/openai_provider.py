import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import (AsyncOpenAI,
                    AsyncStream,
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
from openai.types.chat import ChatCompletionChunk

from seqoria_agent.llm_providers.base_provider import BaseProvider
from seqoria_agent.llm_providers.errors import (ConnectionError,
                                            ContentFilterError,
                                            ContextLengthExceededError,
                                            EnvironmentError,
                                            InvalidRequestError,
                                            LLMError,
                                            ModelNotFoundError,
                                            QuotaExceededError,
                                            ServerError)

from seqoria_agent.llm_providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList
from seqoria_agent.models.llm_models import AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult

_EMPTY_JSON: str = "{}"

class OpenAIProvider(BaseProvider):
    """
    Asynchronous provider implementation for OpenAI's LLM API.
    
    This class provides async methods to generate text completions
    and stream responses from OpenAI's large language models,
    integrating robust error handling and retry mechanisms.
    """
    
    # Default model to use if requested model is unavailable or none is specified
    DEFAULT_MODEL = "gpt-4o" 
    
    def __init__(self, model_id: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize the asynchronous OpenAI provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "gpt-4o", "gpt-3.5-turbo").
                      If None, falls back to DEFAULT_MODEL. Validation occurs during API calls.
            api_key: Explicit API key for authentication. If None, falls back to environment variable.
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
        
        Raises
        ------
            EnvironmentError: If OPENAI_API_KEY environment variable is not set.
            ConnectionError: If the async client cannot be initialized.
        """
        # Prefer the explicitly supplied API key, otherwise fall back to the environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(message="OpenAI API key not provided", missing_vars=["OPENAI_API_KEY"])
        
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
        base_url: Optional[str] = base_url or kwargs.get('base_url') or "https://api.openai.com/v1"
        if base_url:
            client_kwargs['base_url'] = base_url
        
        try:
            # Create and assign the async client
            self.async_client = AsyncOpenAI(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="OpenAI") from e
        
        # Store the requested model_id or use the default if None
        self.model_id = model_id if model_id else self.DEFAULT_MODEL

        # Set default parameters using OpenAI's parameter names
        self.default_params = {}
        
        # Update default_params with any valid query kwargs provided in __init__
        valid_query_keys = {"max_completion_tokens", "temperature", "top_p", "frequency_penalty", 
                            "presence_penalty", "stop", "seed", "response_format", "user", "logit_bias", 
                            "logprobs", "top_logprobs", "audio", "function_call", "functions", "metadata", 
                            "modalities", "n", "parallel_tool_calls", "prediction", "reasoning_effort", 
                            "service_tier", "store", "stream_options", "tool_choice", "tools", 
                            "web_search_options"} # Add other valid keys
        for key, value in kwargs.items():
            if key in valid_query_keys:
                self.default_params[key] = value

    def _prepare_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare request parameters by merging defaults with overrides.
        Removes parameters with None values.

        Args:
            **kwargs: Parameter overrides provided to async_generate/async_stream.
            
        Returns
        -------
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
    
    async def _prepare_chat_history(self, chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]]) -> List[Dict[str, Any]]:
        """
        Normalize `chat_history` so it can be passed directly to the OpenAI Chat API.

        The incoming list can contain a mix of:
        • plain ``dict`` messages using the Seqoria internal schema (``{"role": ..., ...}``)
        • ``ToolCallRequest`` / ``ToolCallResult`` pydantic models
        • ``ChatChunk`` pydantic models (regular user / assistant messages)

        All inputs are converted into the dictionary format expected by the
        OpenAI Chat completions endpoint.
        """

        prepared_chat_history: List[Dict[str, Any]] = []

        for raw_msg in chat_history:
            # Handle *pydantic model* inputs first
            if isinstance(raw_msg, ToolCallRequest):
                prepared_chat_history.append({"role": "assistant",
                                              "tool_calls": [{"id": str(object=raw_msg.id),
                                                              "type": "function",
                                                              "function": {"name": raw_msg.name,
                                                                           "arguments": (_EMPTY_JSON if not raw_msg.arguments else json.dumps(obj=raw_msg.arguments))}}]})
                continue

            if isinstance(raw_msg, ToolCallResult):
                prepared_chat_history.append({"role": "tool",
                                              "content": str(object=raw_msg.result),
                                              "tool_call_id": str(object=raw_msg.tool_call_id)})
                continue

            if isinstance(raw_msg, (AssistantMessage, UserMessage, SystemMessage)):
                prepared_chat_history.append({"role": raw_msg.role,
                                              "content": raw_msg.content})
                continue

            role = raw_msg.get("role")
            if role == "tool_call":
                prepared_chat_history.append({"role": "assistant",
                                              "tool_calls": [{"id": str(object=raw_msg.get("id")),
                                                              "type": "function",
                                                              "function": {"name": raw_msg.get("name"),
                                                                           "arguments": (_EMPTY_JSON if not raw_msg.get("arguments") else json.dumps(obj=raw_msg.get("arguments", {})))}}]})
            elif role == "tool_result":
                prepared_chat_history.append({"role": "tool",
                                              "content": str(object=raw_msg.get("result")),
                                              "tool_call_id": str(object=raw_msg.get("tool_call_id"))})
            elif role in {"assistant", "system", "user"}:
                prepared_chat_history.append({"role": role, 
                                              "content": raw_msg.get("content")})

        return prepared_chat_history
    
    # Reuse shared implementation from BaseProvider (introduced in v0.2)
    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)
    
    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> ToolCall:
        """
        Parse a tool call into a ToolCall object.
        """
        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            try:
                tc["arguments"] = json.loads(s=tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except Exception:
                tc["arguments"] = {}

            processed_tool_calls.append(ToolCall(**tc)) 

        return processed_tool_calls
    
    async def async_generate(self,
                             chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]],
                             tools: Optional[List[Dict[str, Any]]] = None,
                             tool_choice: Optional[Dict[str, Any]] = None,
                             model: Optional[str] = None,
                             **kwargs) -> Union[LLMResponse, ToolCallList]:
        """
        Asynchronously generate a completion for the provided chat history.
        
        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: List of tool dictionaries for tool-based generation in OpenAI format:
                [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            tool_choice: Dictionary specifying the tool choice (e.g., "auto", "none", or {"type": "function", "function": {"name": "..."}}).
            model_id: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).
            
        Returns
        -------
            Union[LLMResponse, ToolCallList]: Response containing:
                - text (str | None): Generated text string. None if generation fails or is empty.
                - usage (LLMUsage): Token usage information
                - tool_calls (list | None): List of tool calls in OpenAI format:
                    [{"id": "call_123", "type": "function", "function": {"name": "...", "arguments": "..."}}]
                - stop_reason (str | None): Reason why generation stopped
            
        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        if chat_history:
            formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        async def _generate() -> Union[LLMResponse, ToolCallList]:
            completion = await self.async_client.chat.completions.create(model=effective_model,
                                                                          messages=formatted_chat_history,
                                                                          stream=False,
                                                                          **params)
            
            content = None
            tool_calls: List[Dict[str, Any]] = []
            
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content
                # Extract tool calls if present - format according to OpenAI specification
                if hasattr(completion.choices[0].message, 'tool_calls') and completion.choices[0].message.tool_calls:
                    for tool_call in completion.choices[0].message.tool_calls:
                        # Attempt to parse the JSON string returned in `arguments`
                        try:
                            parsed_args = json.loads(s=tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                        except Exception:
                            parsed_args = tool_call.function.arguments

                        tool_calls.append(ToolCall(id=tool_call.id,
                                                   type="function_call",
                                                   function_name=tool_call.function.name,
                                                   arguments=parsed_args))

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
            if tool_calls:
                return ToolCallList(model_id=effective_model,
                                    tool_calls=tool_calls,
                                    usage=usage_obj)

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=content,
                               usage=usage_obj,
                               stop_reason=stop_reason)
        
        return await self._execute_with_retry(operation=_generate)
    
    async def async_stream(self, 
                           chat_history: List[Union[Dict[str, Any], "ToolCallRequest", "ToolCallResult", "AssistantMessage", "UserMessage", "SystemMessage"]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Dict[str, Any]] = None,
                           model: Optional[str] = None,
                           **kwargs) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """
        Asynchronously stream a completion for the provided chat history.
        
        Yields content chunks (str) as they arrive, and finally yields a
        LLMResponse containing the full concatenated text and token usage.
        
        Args:
            chat_history: List of message dictionaries.
            tools: List of tool dictionaries for tool-based generation in OpenAI format:
                [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            tool_choice: Dictionary specifying the tool choice (e.g., "auto", "none", or {"type": "function", "function": {"name": "..."}}).
            model_id: Optional identifier for the model to use. If None, uses the default model.
            **kwargs: Generation parameters overriding defaults.
            
        Yields
        ------
            str: Text chunks of the generated completion.
            LLMResponse: Final LLMResponse with "text" (full content), "usage" info, and "tool_calls".
                Example: LLMResponse(model_id="gpt-4o", text="Full response", tool_calls=[...], usage=LLMUsage(...), stop_reason="...")

        Raises
        ------
            Same exceptions as async_generate() method, potentially during iteration.
        """
        params = self._prepare_params(**kwargs)
        
        # Add tools and tool_choice to params if provided
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        if chat_history:
            formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        async def _create_stream() -> AsyncStream[ChatCompletionChunk]:
            # Use the async client here
            return await self.async_client.chat.completions.create(model=effective_model,
                                                                   messages=formatted_chat_history,
                                                                   stream=True,
                                                                   stream_options={"include_usage": True},
                                                                   **params)
        
        full_completion_text = ""
        tool_calls: List[Dict[str, Any]] = []
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
                        yield LLMResponse(model_id=effective_model, 
                                          type="llm_delta", 
                                          text=delta.content,
                                          usage=None,
                                          stop_reason=None)

                     # Handle tool calls - format according to OpenAI specification
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.index is not None:
                                # Ensure we have enough slots in the tool_calls list
                                if len(tool_calls) <= tool_call_delta.index:
                                    tool_calls.append({"id": "",
                                                       "type": "function_call",
                                                       "function_name": "",
                                                       "arguments": ""})
                                 
                                 # Update the tool call at the specified index
                                current_tool_call = tool_calls[tool_call_delta.index]
                                 
                                 # Update ID if provided
                                if not current_tool_call["id"] and tool_call_delta.id:
                                     current_tool_call["id"] = tool_call_delta.id
                                 
                                 # Update type if provided
                                if not current_tool_call["type"] and tool_call_delta.type:
                                    current_tool_call["type"] = tool_call_delta.type
                                 
                                 # Update function details if provided
                                if tool_call_delta.function:
                                    if not current_tool_call["function_name"] and tool_call_delta.function.name:
                                        current_tool_call["function_name"] = tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        current_tool_call["arguments"] += tool_call_delta.function.arguments
                     
                     # Extract finish reason
                    if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                        stop_reason = chunk.choices[0].finish_reason

            usage_obj = LLMUsage(**final_usage_data)
            
            if tool_calls:
                tool_calls_obj = await self._parse_tool_call(tool_call=tool_calls)
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls_obj,
                                   usage=usage_obj)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text,
                                  usage=usage_obj,
                                  stop_reason=stop_reason)

        except Exception as e:
            # Handle errors occurring during async stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error
