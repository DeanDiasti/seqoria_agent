"""
Base provider abstract class for standardizing LLM provider interfaces.

This module defines the BaseProvider abstract base class that all LLM providers
must inherit from, ensuring a consistent interface across different providers
like OpenAI, Anthropic, Bedrock, and SambaNova.

Two variants are provided:
- AsyncBaseProvider: For asynchronous operations (async/await)
- BaseProvider: For synchronous operations (blocking calls)
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from .schemas import LLMResponse, ToolCallList


class AsyncBaseProvider(ABC):
    """
    Abstract base class for all asynchronous LLM providers.

    This class defines the standard interface that all async provider implementations
    must follow, ensuring consistency and type safety across different LLM providers.
    """

    def __init__(self, model_id: Optional[str] = None, **kwargs) -> None:  # noqa: D401
        """No-op default constructor.

        Concrete provider implementations MAY override this to perform real
        client setup.  The default implementation exists so that lightweight
        test stubs (e.g. *DummyProvider* inside the unit-test suite) do **not**
        have to implement an unnecessary ``__init__`` just to satisfy the
        abstract base-class requirements.  Runtime code never relies on any
        state established here.
        """
        self.model_id = model_id

    async def aclose(self) -> None:
        """Release this provider's HTTP client after all requests have finished."""
        client = getattr(self, "async_client", None)
        if client is not None:
            await client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    @abstractmethod
    async def async_generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """
        Asynchronously generate a completion for the provided chat history.

        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: Optional list of tool definitions for function calling.
                   Each tool should have 'name', 'description', and 'input_schema' fields.
            tool_choice: Optional tool choice configuration. Format may vary by provider.
            model: Optional identifier for the model.
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).

        Returns
        -------
            LLMResponse: A standardized response object containing the generated content,
                         tool calls, usage data, and other metadata.

        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        pass

    @abstractmethod
    async def async_stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[LLMResponse, ToolCallList], None]:
        """
        Asynchronously stream a completion for the provided chat history.

        Yields content chunks (str) as they arrive, and finally yields a
        single LLMResponse object containing the full concatenated text,
        tool calls, and token usage.

        Args:
            chat_history: List of message dictionaries.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration.
            model: Optional identifier for the model.
            **kwargs: Generation parameters overriding defaults.

        Yields
        ------
            str: Text chunks of the generated completion.
            LLMResponse: The final, complete response object.

        Raises
        ------
            Same exceptions as async_generate() method, potentially during iteration.
        """
        pass

    # ------------------------------------------------------------------
    # Shared helper – retry wrapper with jittered exponential back-off
    # ------------------------------------------------------------------

    async def _execute_with_retry(self, operation: callable) -> Any:  # noqa: ANN401
        """Execute *operation* with retry/back-off using instance settings.

        Sub-classes must set ``self.max_retries`` **and** implement
        ``self._handle_error(error: Exception) -> Exception`` that converts
        provider-specific exceptions into the canonical :class:`LLMError` tree.
        """
        retries = 0
        last_error: Exception | None = None

        # Fallback if child forgot to set attribute
        max_retries: int = int(getattr(self, "max_retries", 3))

        while retries <= max_retries:
            try:
                return await operation()
            except Exception as exc:  # pragma: no cover – provider-specific
                # Map to our error taxonomy first
                mapped_error = self._handle_error(exc) if hasattr(self, "_handle_error") else exc
                last_error = mapped_error

                # Determine retryability
                from .errors import ConnectionError, ServerError  # local import to avoid cycle

                is_retryable = isinstance(mapped_error, (ConnectionError, ServerError))

                if not is_retryable:
                    raise mapped_error

                retries += 1
                if retries > max_retries:
                    break

                # Exponential back-off with ±10 % jitter
                base = 0.1 * (2 ** (retries - 1))
                import asyncio  # local to minimise cold-start cost
                import random

                await asyncio.sleep(base + random.uniform(0, base * 0.1))

        # Convert into MaxRetriesExceededError for consistency
        from .errors import MaxRetriesExceededError

        op_name = getattr(operation, "__name__", "provider operation")
        raise MaxRetriesExceededError(
            operation=op_name, max_retries=max_retries, last_error=last_error
        )


class BaseProvider(ABC):
    """
    Abstract base class for all synchronous LLM providers.

    This class defines the standard interface that all sync provider implementations
    must follow, ensuring consistency and type safety across different LLM providers.
    """

    def __init__(self, model_id: Optional[str] = None, **kwargs) -> None:  # noqa: D401
        """No-op default constructor.

        Concrete provider implementations MAY override this to perform real
        client setup.  The default implementation exists so that lightweight
        test stubs (e.g. *DummyProvider* inside the unit-test suite) do **not**
        have to implement an unnecessary ``__init__`` just to satisfy the
        abstract base-class requirements.  Runtime code never relies on any
        state established here.
        """
        self.model_id = model_id

    def close(self) -> None:
        """Release this provider's HTTP client."""
        client = getattr(self, "client", None)
        if client is not None:
            client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @abstractmethod
    def generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[LLMResponse, ToolCallList]:
        """
        Synchronously generate a completion for the provided chat history.

        Args:
            chat_history: List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).
            tools: Optional list of tool definitions for function calling.
                   Each tool should have 'name', 'description', and 'input_schema' fields.
            tool_choice: Optional tool choice configuration. Format may vary by provider.
            model: Optional identifier for the model.
            **kwargs: Generation parameters overriding defaults (e.g., temperature, max_tokens).

        Returns
        -------
            LLMResponse: A standardized response object containing the generated content,
                         tool calls, usage data, and other metadata.

        Raises
        ------
            Various LLMError subclasses depending on the API response or connection issues.
        """
        pass

    @abstractmethod
    def stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Union[LLMResponse, ToolCallList], None, None]:
        """
        Synchronously stream a completion for the provided chat history.

        Yields content chunks (str) as they arrive, and finally yields a
        single LLMResponse object containing the full concatenated text,
        tool calls, and token usage.

        Args:
            chat_history: List of message dictionaries.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice configuration.
            model: Optional identifier for the model.
            **kwargs: Generation parameters overriding defaults.

        Yields
        ------
            str: Text chunks of the generated completion.
            LLMResponse: The final, complete response object.

        Raises
        ------
            Same exceptions as generate() method, potentially during iteration.
        """
        pass

    # ------------------------------------------------------------------
    # Shared helper – retry wrapper with jittered exponential back-off
    # ------------------------------------------------------------------

    def _execute_with_retry(self, operation: callable) -> Any:  # noqa: ANN401
        """Execute *operation* with retry/back-off using instance settings.

        Sub-classes must set ``self.max_retries`` **and** implement
        ``self._handle_error(error: Exception) -> Exception`` that converts
        provider-specific exceptions into the canonical :class:`LLMError` tree.
        """
        import time
        import random

        retries = 0
        last_error: Exception | None = None

        # Fallback if child forgot to set attribute
        max_retries: int = int(getattr(self, "max_retries", 3))

        while retries <= max_retries:
            try:
                return operation()
            except Exception as exc:  # pragma: no cover – provider-specific
                # Map to our error taxonomy first
                mapped_error = self._handle_error(exc) if hasattr(self, "_handle_error") else exc
                last_error = mapped_error

                # Determine retryability
                from .errors import ConnectionError, ServerError  # local import to avoid cycle

                is_retryable = isinstance(mapped_error, (ConnectionError, ServerError))

                if not is_retryable:
                    raise mapped_error

                retries += 1
                if retries > max_retries:
                    break

                # Exponential back-off with ±10 % jitter
                base = 0.1 * (2 ** (retries - 1))
                time.sleep(base + random.uniform(0, base * 0.1))

        # Convert into MaxRetriesExceededError for consistency
        from .errors import MaxRetriesExceededError

        op_name = getattr(operation, "__name__", "provider operation")
        raise MaxRetriesExceededError(
            operation=op_name, max_retries=max_retries, last_error=last_error
        )
