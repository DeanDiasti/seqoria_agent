"""AsyncAIClient - MCP-native AI orchestrator (async version).

This module provides the AsyncAIClient class for building AI applications
with tool support via the Model Context Protocol (MCP).

Features:
- Multi-provider LLM support (OpenAI, Anthropic, etc.)
- MCP server connections (HTTP, SSE, stdio)
- Local toolbox support
- Circuit breaker and retry resilience
- Caching for tool results
- Server-populated parameter injection

For synchronous usage, see :class:`AIClient` in ``sync_client.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, Iterable, List, Mapping, Self, Tuple, Union

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.history import build_system_prompt
from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.cache import CacheProtocol, NullCache
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry, ParameterSource
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatChunk,
    LLMUsage,
    ToolCallRequest,
    ToolCallError,
    ToolCallRequestList,
    ToolCallResult,
    ToolMessages,
    UserMessage,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.toolbox import LocalToolbox

__all__ = ["AsyncAIClient"]

HistoryInput = list[Any]

UserContentInput = Union[str, List[Dict[str, Any]]]


class AsyncAIClient:
    """MCP-native AI client for tool-augmented LLM interactions (async version).

    This is the asynchronous version of the AI client. For synchronous usage,
    use :class:`AIClient` instead.

    Parameters
    ----------
    providers : list[AsyncBaseProvider]
        LLM providers to use (required).
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
        Maximum concurrent tool executions across requests, including injected
        parameter resolution and cache I/O.
    tool_timeout : float | None
        Timeout in seconds for parameter resolution, cache I/O, and tool
        execution after acquiring a concurrency slot. Queue time is excluded.
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
    >>> async with AsyncAIClient(providers=[AsyncOpenAIProvider()]) as client:
    ...     async for chunk in client.stream_chat("Hello!", history=[]):
    ...         print(chunk)
    """

    def __init__(
        self,
        providers: List[AsyncBaseProvider] | None = None,
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
        refresh_interval: float | None = None,
        # Caching
        cache: CacheProtocol | None = None,
        cache_ttl: int | None = None,
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
        # Meta-tools
        blocked_tools: Iterable[str] | None = None,
        # Settings
        settings: ClientSettings | None = None,
        # System prompt
        system_prompt: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        if settings is None:
            settings = ClientSettings()

        # Resolve settings with explicit overrides
        providers = settings.providers if providers is None else providers

        # Merge mcp_servers and legacy urls
        _servers: list[str | MCPServerConfig] = []
        if mcp_servers is not None:
            _servers.extend(mcp_servers)
        elif urls is not None:
            _servers.extend(urls)
        elif settings.urls:
            _servers.extend(settings.urls)

        local_toolbox = local_toolbox if local_toolbox is not None else settings.local_toolbox
        logger = logger if logger is not None else settings.logger
        refresh_interval = (
            refresh_interval if refresh_interval is not None else settings.refresh_interval
        )
        cache = cache if cache is not None else settings.cache
        cache_ttl = cache_ttl if cache_ttl is not None else settings.cache_ttl
        max_concurrency = (
            max_concurrency if max_concurrency is not None else settings.max_concurrency
        )
        tool_timeout = tool_timeout if tool_timeout is not None else settings.tool_timeout
        retry_policy = retry_policy if retry_policy is not None else settings.retry_policy
        breaker_threshold = (
            breaker_threshold if breaker_threshold is not None else settings.breaker_threshold
        )
        breaker_reset_after = (
            breaker_reset_after if breaker_reset_after is not None else settings.breaker_reset_after
        )
        param_registry = (
            param_registry
            if param_registry is not None
            else getattr(settings, "param_registry", None)
        )
        blocked_tools = (
            blocked_tools if blocked_tools is not None else getattr(settings, "blocked_tools", ())
        )

        allow_override = settings.allow_override if allow_override is None else allow_override
        if client_kwargs or settings.client_kwargs:
            raise TypeError("Unknown client options; configure transports with MCPServerConfig")

        # Validate
        if not providers:
            raise ValueError("'providers' may not be empty")

        if max_concurrency is not None and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if tool_timeout is not None and (
            not isinstance(tool_timeout, (int, float))
            or isinstance(tool_timeout, bool)
            or not math.isfinite(tool_timeout)
            or tool_timeout <= 0
        ):
            raise ValueError("tool_timeout must be positive and finite")
        if (
            not isinstance(refresh_interval, (int, float))
            or isinstance(refresh_interval, bool)
            or not math.isfinite(refresh_interval)
            or refresh_interval < 0
        ):
            raise ValueError("refresh_interval must be nonnegative and finite")

        if any(not isinstance(provider, AsyncBaseProvider) for provider in providers):
            raise TypeError("AsyncAIClient requires async providers; use an Async*Provider class")

        # Store configuration
        self._providers = list(providers)
        self._servers = list(_servers)
        self._boxes = (
            []
            if local_toolbox is None
            else (local_toolbox if isinstance(local_toolbox, list) else [local_toolbox])
        )
        self._logger = logger or logging.getLogger(name="seqoria_agent")
        self._tool_map: dict[str, Any] = {}

        # Build MCP server configs
        server_configs = self._build_server_configs(_servers)

        # Initialize components
        self.pool = MCPConnectionPool(
            servers=server_configs,
            logger=self._logger,
            retry_policy=retry_policy or {},
            breaker_threshold=breaker_threshold,
            breaker_reset_after=breaker_reset_after,
            tool_map=self._tool_map,
        )

        self.registry = ToolRegistry(
            pool=self.pool,
            local_toolboxes=self._boxes,
            tool_map=self._tool_map,
            blocked_tools=blocked_tools,
            refresh_interval=refresh_interval,
            logger=self._logger,
        )

        self.executor = Executor(
            pool=self.pool,
            registry=self.registry,
            cache=cache or NullCache(),
            cache_ttl=cache_ttl,
            tool_semaphore=(asyncio.Semaphore(value=max_concurrency) if max_concurrency else None),
            tool_timeout=tool_timeout,
            param_registry=param_registry,
            allow_override=allow_override,
            logger=self._logger,
        )

        self.llm = LLMInterface(providers=self._providers, logger=self._logger)

        # System prompt handling
        if system_prompt == "":
            self._system_prompt: str | None = None
        elif system_prompt is None:
            self._system_prompt = build_system_prompt()
        else:
            self._system_prompt = system_prompt.strip()

    def _build_server_configs(
        self,
        servers: list[str | MCPServerConfig],
    ) -> list[MCPServerConfig]:
        """Convert server URLs to MCPServerConfig objects."""
        configs = []

        for server in servers:
            if isinstance(server, MCPServerConfig):
                configs.append(server)
            else:
                configs.append(MCPServerConfig(url=server))

        return configs

    # Context manager methods
    async def __aenter__(self) -> Self:
        try:
            await self.pool.start()
            await self.registry.refresh()
            self.registry.start()
        except BaseException:
            await self.close()
            raise
        if not self.pool.has_servers and not self._boxes:
            self._logger.warning("AsyncAIClient initialized without tools - LLM-only mode")
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        await self.registry.close()
        await self.pool.close()
        self._tool_map.clear()

    # ---------------------------------------------------------------------
    # Dynamic MCP server management
    # ---------------------------------------------------------------------
    async def add_mcp_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        connect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Add MCP servers after initialization.

        - If `connect=True` and the client is already started (inside `async with`),
          this will connect immediately.
        - If `refresh_tools=True`, the ToolRegistry is refreshed so newly available
          tools become visible via `list_tools()` and to the LLM.

        Returns the number of *new* server configs added (deduped by URL).
        """
        # Keep a record of the user's servers list (best-effort, for introspection).
        servers = list(servers)
        self._servers.extend(servers)

        added = await self.pool.add_servers(servers, connect=connect)
        if refresh_tools and added:
            await self.registry.refresh()
        return added

    async def add_mcp_server(
        self,
        server: str | MCPServerConfig,
        *,
        connect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Convenience wrapper for adding a single MCP server."""
        return await self.add_mcp_servers([server], connect=connect, refresh_tools=refresh_tools)

    def list_mcp_servers(self) -> List[MCPServerConfig]:
        """List configured MCP servers (read-only copy)."""
        return self.pool.server_configs

    async def remove_mcp_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        disconnect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Remove MCP servers after initialization.

        - If `disconnect=True`, active connections are closed and routing entries
          for their sessions are pruned.
        - If `refresh_tools=True`, ToolRegistry is refreshed so removed tools stop
          being exposed to the LLM and `list_tools()`.

        Returns number of server configs removed (deduped by URL).
        """
        # Best-effort maintenance of the user-facing servers list
        servers = list(servers)
        urls = {s if isinstance(s, str) else s.url for s in servers}
        if urls:
            self._servers = [
                s for s in self._servers if (s if isinstance(s, str) else s.url) not in urls
            ]

        removed = await self.pool.remove_servers(servers, disconnect=disconnect)
        if refresh_tools and removed:
            await self.registry.refresh()
        return removed

    async def remove_mcp_server(
        self,
        server: str | MCPServerConfig,
        *,
        disconnect: bool = True,
        refresh_tools: bool = True,
    ) -> int:
        """Convenience wrapper for removing a single MCP server."""
        return await self.remove_mcp_servers(
            [server], disconnect=disconnect, refresh_tools=refresh_tools
        )

    # Tool execution methods
    async def call_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        server_args: dict[str, Any] | None = None,
    ) -> List[Any]:
        """Call an MCP tool directly."""
        return await self.pool.call_tool(name=name, arguments=arguments, server_args=server_args)

    async def refresh_tools(self) -> None:
        """Manually refresh the tool registry."""
        await self.registry.refresh()

    async def run_tool_calls(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> List[ToolCallResult | ToolCallError]:
        """Execute tool calls and return results."""
        return await self.executor.run_calls(tool_calls=tool_calls, context=context or {})

    def run_call_stream(
        self,
        tool_calls: List[ToolCallRequest],
        *,
        context: ParamContext | None = None,
    ) -> AsyncGenerator[Tuple[ToolCallRequest, ToolCallResult | ToolCallError], None]:
        """Stream tool call results as they complete."""
        return self.executor.run_call_stream(tool_calls=tool_calls, context=context or {})

    def tools_enabled(self) -> bool:
        """Return True if any tools are available."""
        return bool(self.registry._llm_tools_view())

    def list_tools(self) -> List[dict]:
        """List currently available tools (OpenAI-compatible tool schema).

        Notes
        -----
        - This reflects the most recent `ToolRegistry.refresh()` results.
        - For MCP-backed tools, you typically want to call this after entering the client
          context (`async with client:`) or after calling `await client.refresh_tools()`.
        """
        return self.registry.llm_tools

    # ---------------------------------------------------------------------
    # Tool blocking / filtering
    # ---------------------------------------------------------------------
    async def set_blocked_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Replace the blocked-tool set and optionally refresh the registry."""
        await self.registry.set_blocked_tools(names)
        if refresh:
            await self.registry.refresh()

    async def block_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Add tool names to the blocklist and optionally refresh the registry."""
        await self.registry.block_tools(names)
        if refresh:
            await self.registry.refresh()

    async def unblock_tools(self, names: Iterable[str], *, refresh: bool = True) -> None:
        """Remove tool names from the blocklist and optionally refresh the registry."""
        await self.registry.unblock_tools(names)
        if refresh:
            await self.registry.refresh()

    # LLM interaction methods
    async def llm_generate_once(
        self,
        history: HistoryInput,
        *,
        append_user_message: Union[UserContentInput, None] = None,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Union[AssistantMessage, ToolCallRequestList]:
        """Single LLM generation step."""
        if history is None:
            raise ValueError("history required")
        if not isinstance(history, list):
            raise TypeError("history must be an application-owned list")
        if append_user_message is not None:
            self._ensure_system_prompt(history=history)
            history.append(UserMessage(role="user", content=append_user_message))
        llm_history = self._history_for_llm(history=history, ensure_system_prompt=True)

        if use_tools and self.tools_enabled():
            return await self.llm.generate(
                history=llm_history,
                tools=self.registry._llm_tools_view(),
                tool_choice=tool_choice,
                model=model,
            )
        else:
            return await self.llm.generate(history=llm_history, model=model)

    async def llm_stream_once(
        self,
        history: HistoryInput,
        *,
        append_user_message: Union[UserContentInput, None] = None,
        use_tools: bool = True,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> AsyncGenerator[Union[str, ChatChunk, AssistantMessage, ToolCallRequestList], None]:
        """Stream a single LLM generation step."""
        if history is None:
            raise ValueError("history required")
        if not isinstance(history, list):
            raise TypeError("history must be an application-owned list")
        if append_user_message is not None:
            self._ensure_system_prompt(history=history)
            history.append(UserMessage(role="user", content=append_user_message))
        llm_history = self._history_for_llm(history=history, ensure_system_prompt=True)

        if use_tools and self.tools_enabled():
            async with aclosing(
                self.llm.stream(
                    history=llm_history,
                    tools=self.registry._llm_tools_view(),
                    tool_choice=tool_choice,
                    model=model,
                )
            ) as stream:
                async for chunk in stream:
                    yield chunk
        else:
            async with aclosing(self.llm.stream(history=llm_history, model=model)) as stream:
                async for chunk in stream:
                    yield chunk

    async def chat(
        self,
        user_query: str,
        history: HistoryInput,
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
        user_message_content: Union[UserContentInput, None] = None,
    ) -> AsyncGenerator[Union[AssistantMessage, ToolCallRequestList, ToolMessages], None]:
        """Chat with the LLM, handling tool calls automatically."""
        if history is None:
            raise ValueError("history required")
        if not isinstance(history, list):
            raise TypeError("history must be an application-owned list")
        if user_query is None:
            raise ValueError("user_query required")

        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        tool_ctx: ParamContext = dict(context or {})

        self._append_user_turn(
            history,
            user_query=user_query,
            user_message_content=user_message_content,
        )

        # LLM-only fast path
        if not self.tools_enabled():
            resp = await self.llm_generate_once(history=history, use_tools=False, model=model)
            if isinstance(resp, AssistantMessage):
                history.append(resp)
            yield resp
            return

        total_usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        def _add_usage(u: Any) -> None:
            if u is None:
                return
            total_usage.prompt_tokens += u.prompt_tokens or 0
            total_usage.completion_tokens += u.completion_tokens or 0
            total_usage.total_tokens += u.total_tokens or 0

        for _ in range(max_turns):
            resp = await self.llm_generate_once(
                history=history,
                use_tools=True,
                tool_choice="auto",
                model=model,
            )

            _add_usage(resp.usage)

            if isinstance(resp, AssistantMessage):
                message = AssistantMessage(
                    role="assistant",
                    content=resp.content,
                    thinking=resp.thinking,
                    model=resp.model,
                    usage=total_usage,
                )
                history.append(message)
                yield message
                return

            if isinstance(resp, ToolCallRequestList):
                tool_results = await self.executor.run_calls(
                    tool_calls=resp.tool_calls, context=tool_ctx
                )

                tool_messages = ToolMessages()
                for req, res in zip(resp.tool_calls, tool_results, strict=True):
                    tool_messages.extend([req, res])

                history.extend(tool_messages)
                yield tool_messages

        # Turn limit reached
        turn_limit_message = AssistantMessage(
            role="assistant",
            content="[Turn limit reached]",
            model=None,
            usage=total_usage,
        )
        history.append(turn_limit_message)
        yield turn_limit_message

    async def stream_chat(
        self,
        user_query: str,
        history: HistoryInput,
        *,
        max_turns: int = 5,
        model: str | None = None,
        context: ParamContext | None = None,
        user_message_content: Union[UserContentInput, None] = None,
    ) -> AsyncGenerator[Union[str, ChatChunk, AssistantMessage, ToolMessages], None]:
        """Stream a chat session with the LLM."""
        if history is None:
            raise ValueError("history required")
        if not isinstance(history, list):
            raise TypeError("history must be an application-owned list")
        if user_query is None:
            raise ValueError("user_query required")

        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        tool_ctx: ParamContext = dict(context or {})

        self._append_user_turn(
            history,
            user_query=user_query,
            user_message_content=user_message_content,
        )

        total_usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        def _add_usage(u: Any) -> None:
            if u is None:
                return
            total_usage.prompt_tokens += u.prompt_tokens or 0
            total_usage.completion_tokens += u.completion_tokens or 0
            total_usage.total_tokens += u.total_tokens or 0

        # LLM-only fast path
        if not self.tools_enabled():
            async with aclosing(
                self.llm_stream_once(history=history, use_tools=False, model=model)
            ) as stream:
                async for chunk in stream:
                    if isinstance(chunk, ChatChunk):
                        yield chunk
                        continue
                    if isinstance(chunk, AssistantMessage):
                        _add_usage(chunk.usage)
                        message = AssistantMessage(
                            role="assistant",
                            content=chunk.content,
                            thinking=chunk.thinking,
                            model=chunk.model,
                            usage=total_usage,
                        )
                        history.append(message)
                        yield message
                        return
                    yield chunk
            return

        for _ in range(max_turns):
            async with aclosing(
                self.llm_stream_once(
                    history=history,
                    use_tools=True,
                    tool_choice="auto",
                    model=model,
                )
            ) as stream:
                async for c in stream:
                    if isinstance(c, ChatChunk):
                        yield c
                    else:
                        _add_usage(c.usage)

                        if isinstance(c, AssistantMessage):
                            message = AssistantMessage(
                                role="assistant",
                                content=c.content,
                                thinking=c.thinking,
                                model=c.model,
                                usage=total_usage,
                            )
                            history.append(message)
                            yield message
                            return

                        elif isinstance(c, ToolCallRequestList):
                            tool_results = await self.executor.run_calls(
                                tool_calls=c.tool_calls, context=tool_ctx
                            )

                            tool_messages = ToolMessages()
                            for req, res in zip(c.tool_calls, tool_results, strict=True):
                                tool_messages.extend([req, res])

                            history.extend(tool_messages)
                            yield tool_messages
                            break

                        else:
                            raise ValueError(f"Unknown chunk type: {type(c)}")

        # Turn limit reached
        turn_limit_message = AssistantMessage(
            role="assistant",
            content="[Turn limit reached]",
            model=None,
            usage=total_usage,
        )
        history.append(turn_limit_message)
        yield turn_limit_message

    def _append_user_turn(
        self,
        history: HistoryInput,
        *,
        user_query: str,
        user_message_content: UserContentInput | None,
    ) -> None:
        """Append a user turn to application-owned history."""
        self._ensure_system_prompt(history=history)
        payload = user_message_content if user_message_content is not None else user_query
        history.append(UserMessage(role="user", content=payload))

    def _history_for_llm(
        self,
        history: HistoryInput,
        *,
        ensure_system_prompt: bool,
    ) -> list[Any]:
        """Take a snapshot of application-owned history for this provider call."""
        if ensure_system_prompt:
            self._ensure_system_prompt(history=history)
        return list(history)

    def _ensure_system_prompt(self, history: list[dict[str, Any]]) -> None:
        """Insert system prompt at start of history if not present."""
        if self._system_prompt is None:
            return

        if not history:
            history.insert(0, {"role": "system", "content": self._system_prompt})
            return

        first = history[0]
        if isinstance(first, dict):
            if first.get("role") != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})
        else:
            if getattr(first, "role", None) != "system":
                history.insert(0, {"role": "system", "content": self._system_prompt})
