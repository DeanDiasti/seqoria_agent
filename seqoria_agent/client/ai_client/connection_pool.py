"""MCP Connection Pool with resilience features.

This module manages connections to MCP servers with:
- Multi-transport support (HTTP, SSE, stdio)
- Circuit breaker pattern
- Retry with exponential backoff
- Connection pooling and lifecycle management

Public API
==========
```
async with MCPConnectionPool(servers=["http://localhost:8000/mcp"]) as pool:
    result = await pool.call_tool("my_tool", {"foo": "bar"})
```
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Self

from mcp import ClientSession
from mcp.types import TextContent

from seqoria_agent.client.mcp_transport import (
    MCPServerConfig,
    MCPTransportFactory,
    TransportType,
    HTTPClientPool,
)
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.errors import CircuitOpenError, ToolRemoteError

__all__ = ["MCPConnectionPool", "ConnectionPool"]


@dataclass
class _ActiveConnection:
    """Internal wrapper for an active MCP connection."""

    session: ClientSession
    url: str
    transport_type: TransportType
    _context_manager: Any = field(default=None, repr=False)
    _owner_task: asyncio.Task | None = field(default=None, repr=False)
    _stop_event: asyncio.Event | None = field(default=None, repr=False)


@dataclass(slots=True)
class _TextItem:
    """Lightweight text wrapper matching the `.text` contract used by Executor."""

    text: str


class MCPConnectionPool:
    """Manage MCP server connections with resilience features.

    The pool supports multiple transport types (HTTP, SSE, stdio) and provides:
    - Lazy connection establishment
    - Circuit breaker pattern for fault tolerance
    - Retry with exponential backoff
    - Tool routing to appropriate servers

    Parameters
    ----------
    servers : Iterable[str | MCPServerConfig] | None
        Server URLs or configurations to connect to.
    logger : logging.Logger | None
        Logger instance for connection events.
    retry_policy : dict | None
        Retry configuration with "max_attempts" and "backoff_base" keys.
    breaker_threshold : int
        Number of consecutive failures before circuit trips.
    breaker_reset_after : int
        Seconds before tripped circuit resets.
    tool_map : dict | None
        Mutable mapping of tool_name -> connection (shared with ToolRegistry).
    """

    def __init__(
        self,
        servers: Iterable[str | MCPServerConfig] | None = None,
        *,
        logger: Optional[logging.Logger] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        breaker_threshold: int = 5,
        breaker_reset_after: int = 60,
        tool_map: Optional[Dict[str, Any]] = None,
        http_client_pool: Optional[HTTPClientPool] = None,
    ) -> None:
        # Normalize server configs
        self._server_configs: List[MCPServerConfig] = []
        for server in servers or []:
            if isinstance(server, str):
                self._server_configs.append(MCPServerConfig(url=server))
            else:
                self._server_configs.append(server)

        self._logger = logger or logging.getLogger(name="seqoria_agent.connection_pool")

        # Connection state
        self._lock: asyncio.Lock = asyncio.Lock()
        self._started: bool = False
        self._connections: List[_ActiveConnection] = []
        self._context_managers: List[Any] = []  # Keep context managers alive

        # Shared HTTP client pool for connection reuse
        self._http_client_pool: Optional[HTTPClientPool] = http_client_pool
        self._owns_http_pool: bool = False  # Track if we created the pool

        # Resilience configuration
        self._retry_policy: Dict[str, Any] = retry_policy or {}
        self._breaker_threshold: int = int(breaker_threshold)
        self._breaker_reset_after: int = int(breaker_reset_after)
        self._fail_log: Dict[str, int] = {}
        self._breaker_until: Dict[str, float] = {}

        # Tool routing
        self._tool_map: Dict[str, Any] = tool_map if tool_map is not None else {}
        self._tool_names: dict[str, str] = {}
        self._session_to_url: Dict[ClientSession, str] = {}

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def connections(self) -> List[_ActiveConnection]:
        """Return a copy of the current connection list (read-only)."""
        return list(self._connections)

    @property
    def server_configs(self) -> List[MCPServerConfig]:
        """Return a copy of configured server configs (read-only)."""
        return list(self._server_configs)

    def list_server_urls(self) -> List[str]:
        """Return configured server URLs (read-only)."""
        return [cfg.url for cfg in self._server_configs]

    @property
    def has_servers(self) -> bool:
        """Check whether any remote server connection is alive."""
        return bool(self._connections)

    async def start(self) -> None:
        """Establish connections to all configured servers (idempotent)."""
        async with self._lock:
            # Mark pool as started even if there are no servers yet. This allows
            # dynamic addition of servers after initialization.
            self._started = True

            # Initialize shared HTTP client pool if not provided
            if self._http_client_pool is None and self._server_configs:
                self._http_client_pool = HTTPClientPool()
                self._owns_http_pool = True

            connected_urls = {c.url for c in self._connections}
            missing = [cfg for cfg in self._server_configs if cfg.url not in connected_urls]
            if not missing:
                return

            connect_tasks = [self._connect(config) for config in missing]
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, _ActiveConnection):
                    self._connections.append(result)
                    self._session_to_url[result.session] = result.url
                elif isinstance(result, Exception):
                    self._logger.warning("Connection failed: %s", result)

            if not self._connections and self._server_configs:
                self._logger.warning("MCPConnectionPool initialized without any reachable servers")

    async def add_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        connect: bool = True,
    ) -> int:
        """Add server configs to the pool and optionally connect immediately.

        Returns the number of *new* server configs added (deduped by URL).
        """
        new_configs: list[MCPServerConfig] = []
        for s in servers:
            cfg = MCPServerConfig(url=s) if isinstance(s, str) else s
            new_configs.append(cfg)

        async with self._lock:
            existing_urls = {cfg.url for cfg in self._server_configs}
            to_add = []
            for cfg in new_configs:
                if cfg.url not in existing_urls:
                    to_add.append(cfg)
                    existing_urls.add(cfg.url)
            if not to_add:
                return 0
            self._server_configs.extend(to_add)

        # If pool has been started and caller asked to connect, connect the new ones now.
        if connect and self._started:
            await self.start()

        return len(to_add)

    async def add_server(self, server: str | MCPServerConfig, *, connect: bool = True) -> int:
        """Convenience wrapper for adding a single server. Returns 1 if added, else 0."""
        return await self.add_servers([server], connect=connect)

    def _prune_tool_map_for_session(self, session: Any) -> None:
        """Remove tool_map entries that route to the given session."""
        to_delete = [name for name, target in self._tool_map.items() if target is session]
        for name in to_delete:
            self._tool_map.pop(name, None)

    async def remove_servers(
        self,
        servers: Iterable[str | MCPServerConfig],
        *,
        disconnect: bool = True,
    ) -> int:
        """Remove server configs from the pool and optionally disconnect.

        Returns the number of server configs removed (deduped by URL).
        """
        urls: list[str] = []
        for s in servers:
            urls.append(s if isinstance(s, str) else s.url)

        url_set = {str(u) for u in urls if u}
        if not url_set:
            return 0

        async with self._lock:
            # 1) Remove server configs
            before = len(self._server_configs)
            self._server_configs = [cfg for cfg in self._server_configs if cfg.url not in url_set]
            removed = before - len(self._server_configs)

            # 2) Optionally disconnect active connections
            if disconnect and self._connections:
                remaining: list[_ActiveConnection] = []
                for conn in self._connections:
                    if conn.url not in url_set:
                        remaining.append(conn)
                        continue

                    await self._close_connection(conn)
                    cm = getattr(conn, "_context_manager", None)
                    if cm is not None:
                        try:
                            await cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        try:
                            self._context_managers.remove(cm)
                        except ValueError:
                            pass

                    self._session_to_url.pop(conn.session, None)
                    self._prune_tool_map_for_session(conn.session)

                self._connections = remaining

            return removed

    async def remove_server(
        self,
        server: str | MCPServerConfig,
        *,
        disconnect: bool = True,
    ) -> int:
        """Convenience wrapper for removing a single server. Returns 1 if removed, else 0."""
        return await self.remove_servers([server], disconnect=disconnect)

    async def close(self) -> None:
        """Gracefully shut down all connections."""
        async with self._lock:
            for conn in self._connections:
                await self._close_connection(conn)
            # Legacy/test context managers without an owner task.
            for cm in self._context_managers:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass

            self._connections.clear()
            self._context_managers.clear()
            self._session_to_url.clear()
            self._tool_map.clear()
            self._tool_names.clear()
            self._fail_log.clear()
            self._breaker_until.clear()
            self._started = False
            if self._owns_http_pool and self._http_client_pool is not None:
                await self._http_client_pool.close()
                self._http_client_pool = None
                self._owns_http_pool = False

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], *, server_args: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Invoke a tool on the appropriate server with resilience guards.

        Parameters
        ----------
        name : str
            Tool name to invoke.
        arguments : dict
            Arguments to pass to the tool.
        server_args : dict | None
            Server-populated arguments (injected by executor).

        Returns
        -------
        List[Any]
            List of content items from the tool response.

        Raises
        ------
        ValueError
            If tool is not found in any connected server.
        CircuitOpenError
            If circuit breaker is tripped for the server.
        ToolRemoteError
            If tool execution fails after retries.
        """
        # Resolve the session for this tool
        target = self._tool_map.get(name)
        if target is None:
            raise ValueError(f"Tool '{name}' not found on any connected server.")

        name = self._tool_names.get(name, name)

        # Handle LocalToolbox (non-MCP)

        if isinstance(target, LocalToolbox):
            return await target.call_tool(
                name=name, arguments=arguments, server_args=server_args or {}
            )

        if server_args:
            raise ValueError(
                "Remote MCP server parameters must be resolved on the server from "
                "authenticated request context; client server_args are only supported for local tools"
            )

        # Get server URL for circuit breaker tracking
        # Accept ClientSession or any object with call_tool method (for testing)
        if isinstance(target, ClientSession) or hasattr(target, "call_tool"):
            server_url = self._session_to_url.get(target, "<unknown>")
            session = target
        else:
            raise ValueError(f"Invalid target type for tool '{name}': {type(target)}")

        # Circuit breaker check
        now = time.monotonic()
        if self._breaker_until.get(server_url, 0) > now:
            raise CircuitOpenError(f"Circuit open for server {server_url} - retry later.")

        # Retry policy
        max_attempts = int(self._retry_policy.get("max_attempts", 1))
        backoff_base = float(self._retry_policy.get("backoff_base", 0.5))

        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            try:
                # Call tool via MCP protocol
                result = await session.call_tool(name, arguments)

                # MCP tool errors arrive as results, not transport exceptions.
                if getattr(result, "isError", False) is True:
                    details = "\n".join(
                        item.text for item in result.content if hasattr(item, "text")
                    )
                    raise ToolRemoteError(details or f"Tool '{name}' failed")

                # Success - reset failure counter
                self._fail_log.pop(server_url, None)
                self._breaker_until.pop(server_url, None)

                # Convert MCP result to our format
                return self._convert_tool_result(result)

            except ToolRemoteError:
                # Do not replay a tool that has already run and reported failure.
                raise
            except Exception as exc:
                last_error = exc
                self._fail_log[server_url] = self._fail_log.get(server_url, 0) + 1

                if attempt >= max_attempts:
                    # Trip breaker if threshold met
                    if self._fail_log[server_url] >= self._breaker_threshold:
                        self._breaker_until[server_url] = now + self._breaker_reset_after
                        self._logger.warning("Circuit breaker tripped for %s", server_url)
                    raise exc

                # Exponential backoff before retry
                delay = backoff_base * (2 ** (attempt - 1))
                self._logger.debug("Tool call failed, retrying in %.2fs: %s", delay, exc)
                await asyncio.sleep(delay)

        # Should never reach here
        raise ToolRemoteError(f"Tool call failed after {max_attempts} attempts: {last_error}")

    def _convert_tool_result(self, result: Any) -> List[Any]:
        """Convert MCP CallToolResult to our internal format."""
        contents: List[Any] = []

        if hasattr(result, "content"):
            for item in result.content:
                if isinstance(item, TextContent):
                    contents.append(_TextItem(text=item.text))
                elif hasattr(item, "text"):
                    contents.append(_TextItem(text=item.text))
                else:
                    contents.append(_TextItem(text=str(item)))

        if not contents:
            contents.append(_TextItem(text="Tool executed successfully."))

        return contents

    @staticmethod
    async def _close_connection(conn: _ActiveConnection) -> None:
        if conn._stop_event is not None and conn._owner_task is not None:
            conn._stop_event.set()
            await conn._owner_task

    async def _connect(self, config: MCPServerConfig) -> _ActiveConnection:
        """Keep each SDK transport's task group in its own long-lived owner task."""
        ready = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def own_connection() -> None:
            try:
                async with MCPTransportFactory.connect(
                    config, client_pool=self._http_client_pool
                ) as conn:
                    ready.set_result(conn)
                    await stop.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                elif not isinstance(exc, asyncio.CancelledError):
                    self._logger.warning(
                        "MCP connection closed with error (%s): %s", config.url, exc
                    )

        owner = asyncio.create_task(own_connection())
        try:
            conn = await ready
        except BaseException:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            raise
        self._logger.info("Connected to MCP server: %s", config.url)
        return _ActiveConnection(
            session=conn.session,
            url=config.url,
            transport_type=conn.transport_type,
            _owner_task=owner,
            _stop_event=stop,
        )

    def get_session_for_url(self, url: str) -> Optional[ClientSession]:
        """Get the ClientSession for a specific server URL."""
        for conn in self._connections:
            if conn.url == url:
                return conn.session
        return None


# Backwards compatibility alias
ConnectionPool = MCPConnectionPool
