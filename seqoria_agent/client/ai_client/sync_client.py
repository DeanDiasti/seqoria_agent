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
import threading
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Self,
    Tuple,
    TypeVar,
    Union,
)

from seqoria_agent.client.ai_client.async_client import AsyncAIClient
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry, ParameterSource
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatChunk,
    ToolCallError,
    ToolCallRequest,
    ToolCallRequestList,
    ToolCallResult,
    ToolMessages,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider, BaseProvider, ChatHistory
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList
from seqoria_agent.toolbox import LocalToolbox

__all__ = ["AIClient"]

HistoryInput = list[Any]

UserContentInput = Union[str, List[Dict[str, Any]]]


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get running event loop or create a new one."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    raise RuntimeError(
        "AIClient (sync) cannot be used inside an async context. Use AsyncAIClient instead."
    )


_Result = TypeVar("_Result")


def _run_sync(coro: Coroutine[Any, Any, _Result]) -> _Result:
    """Run a coroutine synchronously."""
    try:
        loop = _get_or_create_event_loop()
    except RuntimeError:
        coro.close()
        raise
    return loop.run_until_complete(coro)


class _SyncProviderAdapter(AsyncBaseProvider):
    """Run blocking providers in worker threads for the sync client facade."""

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    async def async_generate(
        self,
        chat_history: ChatHistory,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse | ToolCallList:
        return await asyncio.to_thread(
            self._provider.generate,
            chat_history=chat_history,
            tools=tools,
            tool_choice=tool_choice,
            model=model,
            **kwargs,
        )

    async def async_stream(
        self,
        chat_history: ChatHistory,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse | ToolCallList, None]:
        stream = self._provider.stream(
            chat_history=chat_history,
            tools=tools,
            tool_choice=tool_choice,
            model=model,
            **kwargs,
        )
        stream_lock = threading.Lock()

        def advance() -> LLMResponse | ToolCallList | None:
            with stream_lock:
                return next(stream, None)

        def close() -> None:
            # Cancelling to_thread does not stop its worker. Serialize close
            # with the pending next() so the generator is never closed while
            # it is executing. The provider's I/O timeout bounds this wait.
            with stream_lock:
                stream.close()

        try:
            while True:
                item = await asyncio.to_thread(advance)
                if item is None:
                    return
                yield item
        finally:
            await asyncio.to_thread(close)


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
        # Local tools
        local_toolbox: LocalToolbox | list[LocalToolbox] | None = None,
        # Logging
        logger: logging.Logger | None = None,
        # Tool registry
        refresh_interval: float | None = None,
        # Execution
        max_concurrency: int | None = None,
        tool_timeout: float | None = None,
        # Resilience
        retry_policy: dict[str, Any] | None = None,
        breaker_threshold: int | None = None,
        breaker_reset_after: int | None = None,
        # Parameters
        param_registry: Union[Mapping[str, ParameterSource], ParameterRegistry, None] = None,
        allow_override: bool | None = None,
        # Settings
        settings: ClientSettings | None = None,
        # System prompt
        system_prompt: str | None = None,
        **unknown_options: Any,
    ) -> None:
        # Convert sync providers to async providers if needed
        async_providers: List[AsyncBaseProvider] = []
        selected_providers = (
            providers if providers is not None else (settings.providers if settings else [])
        )
        if selected_providers:
            for p in selected_providers:
                if isinstance(p, AsyncBaseProvider):
                    async_providers.append(p)
                elif isinstance(p, BaseProvider):
                    async_providers.append(_SyncProviderAdapter(p))
                else:
                    raise TypeError(
                        f"Provider must be BaseProvider or AsyncBaseProvider, got {type(p)}"
                    )

        # Create the underlying async client
        self._async_client = AsyncAIClient(
            providers=async_providers,
            mcp_servers=mcp_servers,
            local_toolbox=local_toolbox,
            logger=logger,
            refresh_interval=refresh_interval,
            max_concurrency=max_concurrency,
            tool_timeout=tool_timeout,
            retry_policy=retry_policy,
            breaker_threshold=breaker_threshold,
            breaker_reset_after=breaker_reset_after,
            param_registry=param_registry,
            allow_override=allow_override,
            settings=settings,
            system_prompt=system_prompt,
            **unknown_options,
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
    # Dynamic MCP server management (sync wrappers)
    # -------------------------------------------------------------------------
    def add_mcp_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        connect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Add MCP servers after initialization (blocking)."""
        return _run_sync(
            self._async_client.add_mcp_servers(
                servers=servers, connect=connect, refresh_tools=refresh_tools
            )
        )

    def add_mcp_server(
        self,
        server: str | MCPServerConfig,
        *,
        connect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Add a single MCP server after initialization (blocking)."""
        return _run_sync(
            self._async_client.add_mcp_server(
                server=server, connect=connect, refresh_tools=refresh_tools
            )
        )

    def list_mcp_servers(self) -> List[MCPServerConfig]:
        """List configured MCP servers (read-only copy)."""
        return self._async_client.list_mcp_servers()

    def remove_mcp_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        disconnect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Remove MCP servers after initialization (blocking)."""
        return _run_sync(
            self._async_client.remove_mcp_servers(
                servers=servers, disconnect=disconnect, refresh_tools=refresh_tools
            )
        )

    def remove_mcp_server(
        self,
        server: str | MCPServerConfig,
        *,
        disconnect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Remove a single MCP server after initialization (blocking)."""
        return _run_sync(
            self._async_client.remove_mcp_server(
                server=server, disconnect=disconnect, refresh_tools=refresh_tools
            )
        )

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
    ) -> List[ToolCallResult | ToolCallError]:
        """Execute tool calls and return results (blocking)."""
        return _run_sync(self._async_client.run_tool_calls(tool_calls=tool_calls, context=context))

    def run_call_stream(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> Generator[Tuple[ToolCallRequest, ToolCallResult | ToolCallError], None, None]:
        """Stream tool call results as they complete (blocking generator)."""
        async_gen = self._async_client.run_call_stream(tool_calls=tool_calls, context=context)

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
    # Tool blocking / filtering (sync wrappers)
    # -------------------------------------------------------------------------
    def set_blocked_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Replace the blocked-tool set and optionally refresh the tool registry."""
        _run_sync(self._async_client.set_blocked_tools(names, refresh=refresh))

    def block_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Add tool names to the blocklist and optionally refresh the tool registry."""
        _run_sync(self._async_client.block_tools(names, refresh=refresh))

    def unblock_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Remove tool names from the blocklist and optionally refresh the tool registry."""
        _run_sync(self._async_client.unblock_tools(names, refresh=refresh))

    # -------------------------------------------------------------------------
    # LLM interaction methods (sync)
    # -------------------------------------------------------------------------
    def llm_generate_once(
        self,
        history: HistoryInput,
        *,
        append_user_message: Union[UserContentInput, None] = None,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Union[AssistantMessage, ToolCallRequestList]:
        """Single LLM generation step (blocking)."""
        return _run_sync(
            self._async_client.llm_generate_once(
                history=history,
                append_user_message=append_user_message,
                use_tools=use_tools,
                tool_choice=tool_choice,
                model=model,
            )
        )

    def llm_stream_once(
        self,
        history: HistoryInput,
        *,
        append_user_message: Union[UserContentInput, None] = None,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Generator[Union[ChatChunk, AssistantMessage, ToolCallRequestList], None, None]:
        """Stream a single LLM generation step (blocking generator)."""
        async_gen = self._async_client.llm_stream_once(
            history=history,
            append_user_message=append_user_message,
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
        history: HistoryInput,
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
        user_message_content: Union[UserContentInput, None] = None,
    ) -> Generator[Union[AssistantMessage, ToolCallRequestList, ToolMessages], None, None]:
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
            user_message_content=user_message_content,
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
        history: HistoryInput,
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
        user_message_content: Union[UserContentInput, None] = None,
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
            user_message_content=user_message_content,
        ):
            if isinstance(item, AssistantMessage):
                final_response = item

        if final_response is None:
            raise RuntimeError("Chat loop completed without assistant response")

        return final_response

    def stream_chat(
        self,
        user_query: str,
        history: HistoryInput,
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
        user_message_content: Union[UserContentInput, None] = None,
    ) -> Generator[Union[ChatChunk, AssistantMessage, ToolMessages], None, None]:
        """Stream a chat session with the LLM (blocking generator).

        Yields chunks as they are received, including tool call results.
        """
        async_gen = self._async_client.stream_chat(
            user_query=user_query,
            history=history,
            max_turns=max_turns,
            model=model,
            context=context,
            user_message_content=user_message_content,
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
