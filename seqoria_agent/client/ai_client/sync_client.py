"""AIClient - MCP-native AI orchestrator (sync version).

This module provides the synchronous AIClient class, a wrapper around
:class:`AsyncAIClient` for blocking/synchronous usage patterns.

For async/await usage, use :class:`AsyncAIClient` directly.

Examples
--------
>>> with AIClient(providers=[OpenAIProvider()]) as client:
...     for chunk in client.stream_chat("Hello!", history=[]):
...         print(chunk)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Generator, Iterable, List, Mapping, Self, Tuple, Union

from seqoria_agent.auth.providers import BaseAuthProvider
from seqoria_agent.client.ai_client.async_client import AsyncAIClient
from seqoria_agent.client.cache import CacheProtocol
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry, ParameterSource
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatChunk,
    ToolCallRequest,
    ToolCallRequestList,
    ToolCallResult,
    ToolMessages,
)
from seqoria_agent.providers.base_provider import BaseProvider, AsyncBaseProvider
from seqoria_agent.toolbox import LocalToolbox

__all__ = ["AIClient"]


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get running event loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - this is problematic for sync wrapper
        raise RuntimeError(
            "AIClient (sync) cannot be used inside an async context. "
            "Use AsyncAIClient instead."
        )
    except RuntimeError:
        # No running loop - create one for sync execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop


def _run_sync(coro):
    """Run a coroutine synchronously."""
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


class AIClient:
    """MCP-native AI client for tool-augmented LLM interactions (sync version).

    This is the synchronous version of the AI client. It wraps :class:`AsyncAIClient`
    and provides blocking methods for all operations.

    For async/await usage, use :class:`AsyncAIClient` instead.

    Parameters
    ----------
    providers : list[BaseProvider | AsyncBaseProvider]
        LLM providers to use (required). Can be sync or async providers.
    mcp_servers : Iterable[str | MCPServerConfig] | None
        MCP servers to connect to (URLs or configs).
    local_toolbox : LocalToolbox | list[LocalToolbox] | None
        Local toolboxes for in-process tool execution.
    logger : logging.Logger | None
        Logger for client events.
    refresh_interval : float
        Interval for automatic tool registry refresh (0 = disabled).
    cache : CacheProtocol | None
        Cache for tool results.
    cache_ttl : int | None
        Cache TTL in seconds.
    max_concurrency : int | None
        Maximum concurrent tool executions.
    tool_timeout : float | None
        Timeout for tool execution in seconds.
    retry_policy : dict | None
        Retry configuration with "max_attempts" and "backoff_base".
    breaker_threshold : int
        Failures before circuit trips.
    breaker_reset_after : int
        Seconds before circuit resets.
    param_registry : Mapping | ParameterRegistry | None
        Server-populated parameter sources.
    allow_override : bool
        Allow LLM to override server-populated params.
    settings : ClientSettings | None
        Default settings (overridden by explicit args).
    auth_provider : BaseAuthProvider | None
        Authentication provider for MCP requests.
    system_prompt : str | None
        System prompt (None = default, "" = disabled).

    Examples
    --------
    >>> with AIClient(providers=[OpenAIProvider()]) as client:
    ...     response = client.chat_sync("Hello!", history=[])
    ...     print(response)

    >>> with AIClient(providers=[OpenAIProvider()]) as client:
    ...     for chunk in client.stream_chat("Hello!", history=[]):
    ...         print(chunk.content, end="")
    """

    def __init__(
        self,
        providers: List[Union[BaseProvider, AsyncBaseProvider]] | None = None,
        *,
        # MCP server connections
        mcp_servers: Iterable[str | MCPServerConfig] | None = None,
        # Legacy alias for mcp_servers
        urls: Iterable[str] | None = None,
        # Local tools
        local_toolbox: LocalToolbox | list[LocalToolbox] | None = None,
        # Logging
        logger: logging.Logger | None = None,
        # Tool registry
        refresh_interval: float = 0,
        # Caching
        cache: CacheProtocol | None = None,
        cache_ttl: int | None = 300,
        # Execution
        max_concurrency: int | None = None,
        tool_timeout: float | None = None,
        # Resilience
        retry_policy: dict[str, Any] | None = None,
        breaker_threshold: int = 5,
        breaker_reset_after: int = 60,
        # Parameters
        param_registry: Union[
            Mapping[str, ParameterSource], ParameterRegistry, None
        ] = None,
        allow_override: bool = False,
        # Settings
        settings: ClientSettings | None = None,
        # Auth
        auth_provider: BaseAuthProvider | None = None,
        # System prompt
        system_prompt: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        # Convert sync providers to async providers if needed
        async_providers: List[AsyncBaseProvider] = []
        if providers:
            for p in providers:
                if isinstance(p, AsyncBaseProvider):
                    async_providers.append(p)
                elif isinstance(p, BaseProvider):
                    # Sync providers have an _async_provider attribute
                    async_providers.append(p._async_provider)
                else:
                    raise TypeError(
                        f"Provider must be BaseProvider or AsyncBaseProvider, got {type(p)}"
                    )

        # Create the underlying async client
        self._async_client = AsyncAIClient(
            providers=async_providers or None,
            mcp_servers=mcp_servers,
            urls=urls,
            local_toolbox=local_toolbox,
            logger=logger,
            refresh_interval=refresh_interval,
            cache=cache,
            cache_ttl=cache_ttl,
            max_concurrency=max_concurrency,
            tool_timeout=tool_timeout,
            retry_policy=retry_policy,
            breaker_threshold=breaker_threshold,
            breaker_reset_after=breaker_reset_after,
            param_registry=param_registry,
            allow_override=allow_override,
            settings=settings,
            auth_provider=auth_provider,
            system_prompt=system_prompt,
            **client_kwargs,
        )

        # Track if we're in context manager
        self._entered = False

    # -------------------------------------------------------------------------
    # Context manager methods (sync)
    # -------------------------------------------------------------------------
    def __enter__(self) -> Self:
        _run_sync(self._async_client.__aenter__())
        self._entered = True
        return self

    def __exit__(self, *args) -> None:
        _run_sync(self._async_client.__aexit__(*args))
        self._entered = False

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        _run_sync(self._async_client.close())

    # -------------------------------------------------------------------------
    # Expose underlying async client for advanced usage
    # -------------------------------------------------------------------------
    @property
    def async_client(self) -> AsyncAIClient:
        """Access the underlying AsyncAIClient for advanced async usage."""
        return self._async_client

    # -------------------------------------------------------------------------
    # Expose internal components (read-only)
    # -------------------------------------------------------------------------
    @property
    def pool(self):
        """Access the MCP connection pool."""
        return self._async_client.pool

    @property
    def registry(self):
        """Access the tool registry."""
        return self._async_client.registry

    @property
    def executor(self):
        """Access the tool executor."""
        return self._async_client.executor

    @property
    def llm(self):
        """Access the LLM interface."""
        return self._async_client.llm

    # -------------------------------------------------------------------------
    # Tool execution methods (sync)
    # -------------------------------------------------------------------------
    def call_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        server_args: dict[str, Any] | None = None,
    ) -> List[Any]:
        """Call an MCP tool directly (blocking)."""
        return _run_sync(
            self._async_client.call_mcp_tool(
                name=name, arguments=arguments, server_args=server_args
            )
        )

    def refresh_tools(self) -> None:
        """Manually refresh the tool registry (blocking)."""
        _run_sync(self._async_client.refresh_tools())

    def run_tool_calls(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> List[ToolCallResult]:
        """Execute tool calls and return results (blocking)."""
        return _run_sync(
            self._async_client.run_tool_calls(tool_calls=tool_calls, context=context)
        )

    def run_call_stream(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> Generator[Tuple[ToolCallRequest, ToolCallResult], None, None]:
        """Stream tool call results as they complete (blocking generator)."""
        async_gen = self._async_client.run_call_stream(
            tool_calls=tool_calls, context=context
        )

        loop = _get_or_create_event_loop()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            # Cleanup the async generator
            loop.run_until_complete(async_gen.aclose())

    def tools_enabled(self) -> bool:
        """Return True if any tools are available."""
        return self._async_client.tools_enabled()

    def list_tools(self) -> List[dict]:
        """List currently available tools (OpenAI-compatible tool schema)."""
        return self._async_client.list_tools()

    # -------------------------------------------------------------------------
    # LLM interaction methods (sync)
    # -------------------------------------------------------------------------
    def llm_generate_once(
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Union[AssistantMessage, ToolCallRequestList]:
        """Single LLM generation step (blocking)."""
        return _run_sync(
            self._async_client.llm_generate_once(
                history=history,
                use_tools=use_tools,
                tool_choice=tool_choice,
                model=model,
            )
        )

    def llm_stream_once(
        self,
        history: List[dict],
        *,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Generator[
        Union[str, ChatChunk, AssistantMessage, ToolCallRequestList], None, None
    ]:
        """Stream a single LLM generation step (blocking generator)."""
        async_gen = self._async_client.llm_stream_once(
            history=history,
            use_tools=use_tools,
            tool_choice=tool_choice,
            model=model,
        )

        loop = _get_or_create_event_loop()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
    ) -> Generator[
        Union[AssistantMessage, ToolCallRequestList, ToolMessages], None, None
    ]:
        """Chat with the LLM, handling tool calls automatically (blocking generator).

        This method yields responses as they are generated, handling tool calls
        automatically in the background.
        """
        async_gen = self._async_client.chat(
            user_query=user_query,
            history=history,
            max_turns=max_turns,
            model=model,
            context=context,
        )

        loop = _get_or_create_event_loop()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def chat_sync(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
    ) -> AssistantMessage:
        """Chat with the LLM and return final response (blocking, non-streaming).

        This is a convenience method that consumes the entire chat loop and
        returns only the final assistant message.
        """
        final_response: AssistantMessage | None = None

        for item in self.chat(
            user_query=user_query,
            history=history,
            max_turns=max_turns,
            model=model,
            context=context,
        ):
            if isinstance(item, AssistantMessage):
                final_response = item

        if final_response is None:
            raise RuntimeError("Chat loop completed without assistant response")

        return final_response

    def stream_chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
    ) -> Generator[
        Union[str, ChatChunk, AssistantMessage, ToolMessages], None, None
    ]:
        """Stream a chat session with the LLM (blocking generator).

        Yields chunks as they are received, including tool call results.
        """
        async_gen = self._async_client.stream_chat(
            user_query=user_query,
            history=history,
            max_turns=max_turns,
            model=model,
            context=context,
        )

        loop = _get_or_create_event_loop()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def add_auth_provider(self, provider: BaseAuthProvider) -> None:
        """Replace the authentication provider for MCP requests."""
        self._async_client.add_auth_provider(provider)

