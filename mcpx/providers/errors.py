# core/llm/errors.py

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
    
    def __init__(self, model_id, provider, fallback_model=None) -> None:
        """
        Initialize the error with model information.
        
        Args:
            model_id: ID of the requested model that wasn't found
            provider: Name of the provider
            fallback_model: Model ID being used as fallback, if any
        """
        self.model_id = model_id
        self.provider = provider
        self.fallback_model = fallback_model
        message = f"Model '{model_id}' not found or not accessible on {provider}"
        if fallback_model:
            message += f". Using fallback model '{fallback_model}'"
        super().__init__(message)


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