import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Tuple

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
from .base_provider import BaseProvider
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

_EMPTY_JSON: str = "{}"

class AnthropicProvider(BaseProvider):
    """
    Asynchronous provider implementation for Anthropic's Claude API.
    
    This class provides async methods to generate text completions
    and stream responses from Anthropic's Claude models,
    integrating robust error handling and retry mechanisms.
    Supports tool usage and function calling.
    """
    
    # Default model to use if requested model is unavailable or none is specified
    DEFAULT_MODEL = "claude-sonnet-4-20250514" 
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the asynchronous Anthropic provider.
        
        Args:
            model_id: Identifier for the model to use (e.g., "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307").
                      If None, falls back to DEFAULT_MODEL. Validation occurs during API calls.
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
                    # Add other valid Anthropic parameters as needed
                Client kwargs (used for client setup):
                    - timeout (float): Request timeout in seconds (default: 30.0)
                    - max_retries (int): Maximum number of retries for retryable errors (default: 3)
        
        Raises
        ------
            EnvironmentError: If ANTHROPIC_API_KEY environment variable is not set.
            ConnectionError: If the async client cannot be initialized.
        """
        # Prefer the explicitly supplied API key, otherwise fall back to the environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise EnvironmentError(message="Anthropic API key not provided", missing_vars=["ANTHROPIC_API_KEY"])
        
        # Store max_retries for use in retry logic
        self.max_retries = kwargs.get('max_retries', 3)
        
        # Initialize only the asynchronous Anthropic client
        client_kwargs = {'timeout': kwargs.get('timeout', 30.0),
                         'max_retries': 0} # Disable automatic retries in the client; handled manually
        
        try:
            # Create and assign the async client
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)
        except Exception as e:
            # Catch any exception during initialization
            raise ConnectionError(provider="Anthropic") from e
        
        # Store the requested model_id or use the default if None
        self.model_id = model_id if model_id else self.DEFAULT_MODEL

        # Set default parameters using Anthropic's parameter names
        self.default_params = {"max_tokens": 1024,
                               "temperature": 0.7,
                               "top_p": 1.0}
        
        # Update default_params with any valid query kwargs provided in __init__
        valid_query_keys = self.default_params.keys() | {"top_k", "stop_sequences", "metadata", "tools", "tool_choice"} # Add other valid keys
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
             return ModelNotFoundError(model_id=self.model_id, provider="Anthropic", fallback_model=self.DEFAULT_MODEL)
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
        elif isinstance(error, asyncio.TimeoutError) or "timed out" in str(object=error).lower(): # Catch asyncio timeouts
            return ConnectionError(provider="Anthropic", message="Request timed out")
            
        # Fallback for unexpected errors
        return LLMError(message=f"An unexpected Anthropic error occurred: {error}")
    
    # Reuse shared implementation from BaseProvider (introduced in v0.2)
    async def _execute_with_retry(self, operation: callable) -> Any:  # type: ignore[override]
        return await super()._execute_with_retry(operation=operation)
    
    async def _parse_tool_call(self, tool_call: List[Dict[str, Any]]) -> List[ToolCall]:  # noqa: D401
        """Convert *tool_call* dictionaries into :class:`ToolCall` objects.

        Ensures the *arguments* field is a mapping (JSON-decoded if it was
        received as a JSON string) so it satisfies the schema.
        """

        processed_tool_calls: List[ToolCall] = []

        for tc in tool_call:
            # Decode argument payload if it is a JSON string
            if isinstance(tc.arguments, str):
                try:
                    tc.arguments = json.loads(s=tc.arguments)
                except Exception:  # noqa: BLE001 – fall back to empty mapping
                    tc.arguments = {}

            processed_tool_calls.append(ToolCall(**tc))

        return processed_tool_calls
    
    async def _format_tool_call(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the tool calls into the format expected by the Anthropic API.
        """
        return [{"name": str(object=tool["function"]["name"]), 
                 "description": tool["function"]["description"], 
                 "input_schema": tool["function"]["parameters"]} for tool in tools]

    async def _prepare_chat_history(self,
                                    chat_history: List[Dict[str, Any]],
                                    *,
                                    enable_cache: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare the chat history for the Anthropic API.

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
        """

        prepared_chat_history: List[Dict[str, Any]] = []
        system_message: List[Dict[str, Any]] = []

        for message in chat_history:
            if isinstance(message, ToolCallRequest):
                prepared_chat_history.append({"role": "assistant",
                                              "tool_calls": [{"id": message.id,
                                                              "type": "function",
                                                              "function": {"name": message.name,
                                                                           "arguments": (_EMPTY_JSON if not message.arguments else json.dumps(obj=message.arguments))}}]})
                continue

            if isinstance(message, ToolCallResult):
                prepared_chat_history.append({"role": "tool",
                                              "content": message.result,
                                              "tool_call_id": message.tool_call_id})
                continue

            if isinstance(message, (AssistantMessage, UserMessage)):
                prepared_chat_history.append({"role": message.role,
                                              "content": message.content})
                continue

            if isinstance(message, SystemMessage):
                sys_block = {"type": "text", 
                             "text": message.content}
                if enable_cache:
                    sys_block["cache_control"] = {"type": "ephemeral"}

                system_message.append(sys_block)

            role = message.get("role")
            if role == "tool_call":
                prepared_chat_history.append({"role": "user", "content": [{"type": "tool_use",
                                                                           "name": message.get("name"),
                                                                           "id": message.get("id"),
                                                                           "input": (_EMPTY_JSON if not message.get("arguments") else json.dumps(obj=message.get("arguments")))}]})
                
            elif role == "tool_result":
                prepared_chat_history.append({"role": "user", "content": [{"tool_use_id": message.get("tool_call_id"),
                                                                           "type": "tool_result",
                                                                           "content": message.get("result")}]})
                
            elif role in {"assistant", "user"}:    
                prepared_chat_history.append({"role": role, 
                                              "content": [{"type": "text",
                                                           "text": message.get("content")}]})

            if message.get("role") == "system":
                sys_block = {"type": "text", "text": message["content"]}
                if enable_cache:
                    sys_block["cache_control"] = {"type": "ephemeral"}

                system_message.append(sys_block)

        if enable_cache and not system_message and prepared_chat_history:
            first_msg = prepared_chat_history[0]
            # We can only add cache_control on rich content blocks (list form)
            content = first_msg.get("content")
            if isinstance(content, list) and content:
                # Attach marker to the final block of the list
                content[-1]["cache_control"] = {"type": "ephemeral"}

        return system_message, prepared_chat_history

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
            params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice

        if chat_history:
            system_message, formatted_chat_history = await self._prepare_chat_history(chat_history=chat_history)

        effective_model = model or self.model_id
        async def _generate() -> Union[LLMResponse, ToolCallList]:
            
            completion = await self.async_client.messages.create(model=effective_model,
                                                                 messages=formatted_chat_history,
                                                                 system=system_message,
                                                                 **params)
            
            # Extract text content and tool calls
            text_content = None
            tool_calls = []
            
            if completion.content and len(completion.content) > 0:
                for content_block in completion.content:
                    if hasattr(content_block, 'text') and content_block.type == 'text':
                        # Concatenate text blocks (there might be multiple)
                        if text_content is None:
                            text_content = content_block.text
                        else:
                            text_content += content_block.text
                    elif hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        try:
                            parsed_args = json.loads(s=content_block.input) if isinstance(content_block.input, str) else content_block.input
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

            return LLMResponse(model_id=effective_model,
                               type="llm_response",
                               text=text_content,
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
            params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice

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
        
        full_completion_text = ""
        tool_calls: List[Dict[str, Any]] = []
        current_tool_call = None
        current_tool_input_json = ""
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
                        final_usage_data["prompt_tokens"] = chunk.message.usage.input_tokens or 0
                elif chunk.type == "content_block_start":
                    # Handle start of content blocks (text or tool_use)
                    if hasattr(chunk, 'content_block'):
                        if chunk.content_block.type == 'tool_use':
                            # Start of a tool use block
                            current_tool_call = ToolCall(id=chunk.content_block.id,
                                                         type="tool_use",
                                                         function_name=chunk.content_block.name,
                                                         arguments=chunk.content_block.input or {})  # May be empty initially
                            current_tool_input_json = ""  # Reset for accumulating JSON
                elif chunk.type == "content_block_delta":
                    # Extract text delta from content block delta
                    if hasattr(chunk, 'delta'):
                        if hasattr(chunk.delta, 'text'):
                            # Text content delta
                            delta_text = chunk.delta.text
                            if delta_text:
                                full_completion_text += delta_text
                                yield LLMResponse(model_id=effective_model, 
                                                  type="llm_delta", 
                                                  text=delta_text,
                                                  usage=None,
                                                  stop_reason=None)
                                
                        elif hasattr(chunk.delta, 'partial_json') and current_tool_call:
                            # Tool use input delta (partial JSON)
                            # Accumulate the partial JSON strings
                            current_tool_input_json += chunk.delta.partial_json
                elif chunk.type == "content_block_stop":
                    # End of a content block
                    if current_tool_call:
                        # Complete the tool call and add it to the list
                        try:
                            # Parse the accumulated JSON if we have it
                            if current_tool_input_json:
                                current_tool_call["arguments"] = json.loads(s=current_tool_input_json)
                            # If no accumulated JSON, use what was in the start block
                            tool_calls.append(current_tool_call)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, use the input from the start block
                            tool_calls.append(current_tool_call)
                        
                        current_tool_call = None
                        current_tool_input_json = ""
                elif chunk.type == "message_delta":
                    # Extract final usage and stop reason from message delta event
                    if hasattr(chunk, 'usage'):
                        final_usage_data.completion_tokens = chunk.usage.output_tokens or 0
                        final_usage_data.total_tokens = final_usage_data.prompt_tokens + final_usage_data.completion_tokens
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                        stop_reason = chunk.delta.stop_reason
            

            if tool_calls:
                tool_calls_obj = await self._parse_tool_call(tool_call=tool_calls)
                yield ToolCallList(model_id=effective_model,
                                   tool_calls=tool_calls_obj,
                                   usage=final_usage_data)
            else:
                yield LLMResponse(model_id=effective_model,
                                  type="llm_response",
                                  text=full_completion_text,
                                  usage=final_usage_data,
                                  stop_reason=stop_reason)

        except Exception as e:
            # Handle errors occurring during async stream iteration
            mapped_error = self._handle_error(error=e)
            raise mapped_error