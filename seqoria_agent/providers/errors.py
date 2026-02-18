# core/llm/errors.py
"""LLM Provider Error Hierarchy and Unified Error Mapping.

This module defines the exception hierarchy for LLM provider errors
and provides utilities for mapping provider-specific exceptions to
our standardized error types.
"""

from __future__ import annotations

import asyncio
import re
from typing import Callable, Dict, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class LLMError(Exception):
    """Base class for all LLM-related errors."""

    def __init__(self, message="An error occurred", provider=None):
        self.provider = provider
        super().__init__(message)


class EnvironmentError(LLMError):
    """Raised when required environment variables are missing."""
    
    def __init__(self, message="Required environment variables are missing", missing_vars=None) -> None:
        """
        Initialize the error with detailed information.
        
        Args:
            message: Error description message
            missing_vars: List of missing environment variable names
        """
        self.missing_vars = missing_vars or []
        super().__init__(f"{message}: {', '.join(self.missing_vars)}" if self.missing_vars else message)


class ConnectionError(LLMError):
    """Raised when connection to the LLM provider fails."""
    
    def __init__(self, provider, message="Cannot connect to LLM provider") -> None:
        """
        Initialize the error with provider information.
        
        Args:
            provider: Name of the provider that couldn't be connected to
            message: Error description message
        """
        self.provider = provider
        super().__init__(f"{message}: {provider}")


class ModelNotFoundError(LLMError):
    """Raised when the requested model doesn't exist or isn't accessible."""
    
    def __init__(self, model_id, provider, fallback_model=None, available_models=None) -> None:
        """
        Initialize the error with model information.
        
        Args:
            model_id: ID of the requested model that wasn't found
            provider: Name of the provider
            fallback_model: Model ID being used as fallback, if any
            available_models: List of available model IDs from the provider
        """
        self.model_id = model_id
        self.fallback_model = fallback_model
        self.available_models = available_models or []
        
        message = f"Model '{model_id}' not found or not accessible on {provider}"
        if fallback_model:
            message += f". Using fallback model '{fallback_model}'"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, provider=provider)


class ContextLengthExceededError(LLMError):
    """Raised when prompt exceeds model's context window."""
    
    def __init__(self, model_id, tokens=None, max_tokens=None) -> None:
        """
        Initialize the error with context length details.
        
        Args:
            model_id: ID of the model whose context length was exceeded
            tokens: Number of tokens in the prompt (if known)
            max_tokens: Maximum allowed tokens (if known)
        """
        self.model_id = model_id
        self.tokens = tokens
        self.max_tokens = max_tokens
        
        message = f"Prompt exceeds maximum context length for model '{model_id}'"
        if tokens and max_tokens:
            message += f": {tokens} tokens (max: {max_tokens})"
        super().__init__(message)


class ContentFilterError(LLMError):
    """Raised when content is filtered for safety reasons."""
    
    def __init__(self, provider, filter_reason=None) -> None:
        """
        Initialize the error with filter information.
        
        Args:
            provider: Name of the provider that filtered the content
            filter_reason: Reason for filtering, if provided
        """
        self.provider = provider
        self.filter_reason = filter_reason
        
        message = f"Content filtered by {provider}"
        if filter_reason:
            message += f": {filter_reason}"
        super().__init__(message)


class MaxRetriesExceededError(LLMError):
    """Raised when maximum retry attempts have been exceeded."""
    
    def __init__(self, operation, max_retries, last_error=None) -> None:
        """
        Initialize the error with retry information.
        
        Args:
            operation: Description of the operation that failed
            max_retries: Maximum number of retries that were attempted
            last_error: The last error encountered (if available)
        """
        self.operation = operation
        self.max_retries = max_retries
        self.last_error = last_error
        
        message = f"Maximum retries ({max_retries}) exceeded for operation: {operation}"
        if last_error:
            message += f". Last error: {str(object=last_error)}"
        super().__init__(message)


class InvalidRequestError(LLMError):
    """Raised when request parameters are invalid."""
    
    def __init__(self, message, param=None) -> None:
        """
        Initialize the error with parameter details.
        
        Args:
            message: Error description message
            param: Name of the invalid parameter, if specific
        """
        self.param = param
        if param:
            message = f"Invalid parameter '{param}': {message}"
        super().__init__(message)


class QuotaExceededError(LLMError):
    """Raised when provider quota or rate limit is exceeded."""
    
    def __init__(self, provider, limit_type="quota") -> None:
        """
        Initialize the error with quota information.
        
        Args:
            provider: Name of the provider whose quota was exceeded
            limit_type: Type of limit exceeded (e.g., "quota", "rate", "budget")
        """
        self.provider = provider
        self.limit_type = limit_type
        super().__init__(f"{limit_type.capitalize()} exceeded for provider {provider}")


class ServerError(LLMError):
    """Raised when the provider experiences an internal server error."""
    
    def __init__(self, provider, status_code=None) -> None:
        """
        Initialize the error with server information.
        
        Args:
            provider: Name of the provider experiencing the error
            status_code: HTTP status code if applicable
        """
        self.provider = provider
        self.status_code = status_code
        
        message = f"Server error on {provider}"
        if status_code:
            message += f" (status code: {status_code})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Unified Error Mapper
# ---------------------------------------------------------------------------

class ErrorMapper:
    """Unified error mapper for converting provider-specific exceptions.
    
    This class provides a declarative way to map provider exceptions to
    our standardized error hierarchy, reducing boilerplate in providers.
    
    Example
    -------
    >>> class AuthenticationError(Exception): ...
    >>> class RateLimitError(Exception): ...
    >>> mapper = ErrorMapper("OpenAI")
    >>> mapper.register(AuthenticationError, 
    ...     lambda e, m: EnvironmentError(f"Auth failed: {e}", missing_vars=["OPENAI_API_KEY"]))
    >>> mapper.register(RateLimitError, 
    ...     lambda e, m: QuotaExceededError(provider="OpenAI", limit_type="rate"))
    >>> 
    >>> # In provider code:
    >>> try:
    ...     result = await client.chat.create(...)
    >>> except Exception as e:
    ...     raise mapper.map(e, model_id="gpt-4o")
    """
    
    def __init__(self, provider_name: str) -> None:
        """Initialize the error mapper.
        
        Parameters
        ----------
        provider_name : str
            Name of the provider (for error messages).
        """
        self.provider_name = provider_name
        self._mappings: list[Tuple[Type[Exception], Callable[[Exception, str | None], LLMError]]] = []
        self._status_code_handlers: Dict[int, Callable[[Exception, str | None], LLMError]] = {}
        self._pattern_handlers: list[Tuple[re.Pattern, Callable[[Exception, str | None, re.Match], LLMError]]] = []
    
    def register(
        self,
        exc_type: Type[Exception],
        handler: Callable[[Exception, str | None], LLMError],
    ) -> "ErrorMapper":
        """Register a handler for a specific exception type.
        
        Parameters
        ----------
        exc_type : Type[Exception]
            The exception type to handle.
        handler : Callable
            Function that takes (exception, model_id) and returns an LLMError.
            
        Returns
        -------
        ErrorMapper
            Self for chaining.
        """
        self._mappings.append((exc_type, handler))
        return self
    
    def register_status_code(
        self,
        status_code: int,
        handler: Callable[[Exception, str | None], LLMError],
    ) -> "ErrorMapper":
        """Register a handler for a specific HTTP status code.
        
        Parameters
        ----------
        status_code : int
            The HTTP status code to handle.
        handler : Callable
            Function that takes (exception, model_id) and returns an LLMError.
            
        Returns
        -------
        ErrorMapper
            Self for chaining.
        """
        self._status_code_handlers[status_code] = handler
        return self
    
    def register_pattern(
        self,
        pattern: str,
        handler: Callable[[Exception, str | None, re.Match], LLMError],
    ) -> "ErrorMapper":
        """Register a handler for exceptions matching a regex pattern.
        
        Parameters
        ----------
        pattern : str
            Regex pattern to match against exception message.
        handler : Callable
            Function that takes (exception, model_id, match) and returns an LLMError.
            
        Returns
        -------
        ErrorMapper
            Self for chaining.
        """
        self._pattern_handlers.append((re.compile(pattern, re.IGNORECASE), handler))
        return self
    
    def map(self, error: Exception, model_id: str | None = None) -> LLMError:
        """Map an exception to our error hierarchy.
        
        Parameters
        ----------
        error : Exception
            The exception to map.
        model_id : str | None
            The model ID for context (used in some error messages).
            
        Returns
        -------
        LLMError
            The mapped error.
        """
        # Check for timeout first (common across all providers)
        if isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).lower():
            return ConnectionError(provider=self.provider_name, message="Request timed out")
        
        # Check registered exception types
        for exc_type, handler in self._mappings:
            if isinstance(error, exc_type):
                return handler(error, model_id)
        
        # Check status code handlers (for API errors)
        status_code = getattr(error, "status_code", None)
        if status_code is not None and status_code in self._status_code_handlers:
            return self._status_code_handlers[status_code](error, model_id)
        
        # Check pattern handlers
        error_message = str(error)
        for pattern, handler in self._pattern_handlers:
            match = pattern.search(error_message)
            if match:
                return handler(error, model_id, match)
        
        # Handle generic status codes
        if status_code is not None:
            if 500 <= status_code < 600:
                return ServerError(provider=self.provider_name, status_code=status_code)
            elif status_code == 429:
                return QuotaExceededError(provider=self.provider_name, limit_type="rate")
            elif status_code == 400:
                return InvalidRequestError(message=f"API Error {status_code}: {error}")
        
        # Fallback to generic error
        return LLMError(message=f"An unexpected {self.provider_name} error occurred: {error}")
    
    def __call__(self, error: Exception, model_id: str | None = None) -> LLMError:
        """Callable shorthand for map()."""
        return self.map(error, model_id)


def create_openai_error_mapper() -> ErrorMapper:
    """Create an error mapper pre-configured for OpenAI.
    
    Returns
    -------
    ErrorMapper
        Configured mapper for OpenAI exceptions.
    """
    mapper = ErrorMapper("OpenAI")
    
    # We can't import OpenAI exceptions here (circular/optional dep)
    # Instead, providers should call this and add their mappings
    
    # Register common patterns
    mapper.register_pattern(
        r"context_length_exceeded",
        lambda e, m, _: ContextLengthExceededError(model_id=m or "unknown"),
    )
    mapper.register_pattern(
        r"content_policy_violation",
        lambda e, m, _: ContentFilterError(provider="OpenAI", filter_reason=str(e)),
    )
    
    return mapper


def create_anthropic_error_mapper() -> ErrorMapper:
    """Create an error mapper pre-configured for Anthropic.
    
    Returns
    -------
    ErrorMapper
        Configured mapper for Anthropic exceptions.
    """
    mapper = ErrorMapper("Anthropic")
    
    # Register common patterns
    mapper.register_pattern(
        r"context_length_exceeded|maximum context length",
        lambda e, m, _: ContextLengthExceededError(model_id=m or "unknown"),
    )
    mapper.register_pattern(
        r"content_policy_violation|safety",
        lambda e, m, _: ContentFilterError(provider="Anthropic", filter_reason=str(e)),
    )
    mapper.register_pattern(
        r"tool.*(?:invalid|missing)",
        lambda e, m, _: InvalidRequestError(message=f"Tool validation error: {e}"),
    )
    
    return mapper