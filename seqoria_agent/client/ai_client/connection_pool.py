from __future__ import annotations

"""Connection management utilities for *seqoria_agent*'s new AIClient implementation.

The pool is responsible _only_ for maintaining network connections to one or more
FastMCP servers and providing a thin, retry-aware wrapper around the ``tools/call``
JSON-RPC method.  All higher-level concerns (tool registry, caching, meta-tools, …)
remain outside the scope of this component.

Public API
==========
```
async with ConnectionPool(urls=[...]) as pool:
    result = await pool.call_tool("my_tool", {"foo": "bar"})
```

The class purposefully exposes a **narrow** interface so later phases can swap the
transport or add more sophisticated load-balancing without affecting callers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Self

from seqoria_agent.models.errors import CircuitOpenError, ToolRemoteError  # noqa: F401
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.simple_http_transport import SimpleHTTPTransport  # type: ignore

__all__: list[str] = ["ConnectionPool"]


class _Connection(NamedTuple):
    client: SimpleHTTPTransport
    url: str

class ConnectionPool:
    """Maintain HTTP connections + retry / circuit-breaker resiliency logic.

    The pool is configured via constructor but _lazy-connects_ in :pyfunc:`start` so
    expensive network hand-shakes only happen when explicitly required.
    """

    def __init__(self,
                 urls: Iterable[str] | None = None,
                 *,
                 logger: Optional[logging.Logger] = None,
                 retry_policy: Optional[Dict[str, Any]] = None,
                 breaker_threshold: int = 5,
                 breaker_reset_after: int = 60,
                 transport_cls: type[SimpleHTTPTransport] = SimpleHTTPTransport,
                 tool_map: Optional[Dict[str, Any]] = None,
                 client_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self._urls: List[str] = list(dict.fromkeys(urls or []))
        self._logger = logger or logging.getLogger(name="seqoria_agent.connection_pool")

        self._lock: asyncio.Lock = asyncio.Lock()
        self._connections: List[_Connection] = []
        self._transport_cls: type[SimpleHTTPTransport] = transport_cls
        self._client_kwargs: Dict[str, Any] = client_kwargs or {}

        self._retry_policy: Dict[str, Any] = retry_policy or {}
        self._breaker_threshold: int = int(breaker_threshold)
        self._breaker_reset_after: int = int(breaker_reset_after)
        self._fail_log: Dict[str, int] = {}
        self._breaker_until: Dict[str, float] = {}

        self._tool_map: Dict[str, Any] = tool_map if tool_map is not None else {}
        self._client_to_url: Dict[SimpleHTTPTransport, str] = {}

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def connections(self) -> List[_Connection]:
        """Return a *copy* of the current connection list (read-only helper)."""
        return list(self._connections)

    @property
    def has_servers(self) -> bool:  # noqa: D401 – keep short
        """Quick check whether *any* remote server connection is alive."""
        return bool(self._connections)

    async def start(self) -> None:  # noqa: D401 – simple coro
        """Establish connections to all configured URLs (idempotent)."""
        async with self._lock:
            if self._connections:  # Already connected
                return

            connect_tasks = [self._connect(url=url) for url in self._urls]
            results = await asyncio.gather(*connect_tasks)
            self._connections = [c for c in results if c is not None]

            # Build reverse lookup mapping in one go (faster than per-call scan)
            self._client_to_url = {conn.client: conn.url for conn in self._connections}

            if not self._connections:
                self._logger.warning("ConnectionPool initialized without any reachable servers")

    async def close(self) -> None:  # noqa: D401 – simple coro
        """Gracefully shut down all HTTP transports and reset state."""
        async with self._lock:
            # Close transports sequentially – AsyncClient.aclose already avoids
            # expensive awaits when connections are idle; gather() adds overhead.
            for conn in self._connections:
                try:
                    await conn.client.close()
                except Exception:
                    # Ignore individual close errors – pool is shutting down
                    continue
            self._connections.clear()
            self._client_to_url.clear()
            # Leave breaker + fail logs intact - they are per URL and still
            # relevant if the app re-connects later in its lifetime.

    # --------------------------------------------------------------
    # Tool invocation (remote OR local via tool_map)
    # --------------------------------------------------------------

    async def call_tool(self,
                        name: str,
                        arguments: Dict[str, Any],
                        *,
                        server_args: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Invoke *name* on the appropriate server with retry & breaker logic."""
        # Resolve the transport based on registration done by ToolRegistry/AIClient
        client = self._tool_map.get(name)
        if client is None:
            raise ValueError(f"Tool '{name}' not found on any connected server.")

        if isinstance(client, LocalToolbox):
            return await client.call_tool(name=name, 
                                          arguments=arguments, 
                                          server_args=server_args or {})

        # Remote call with resilience guards
        server_url = self._find_server_url(client=client)
        if server_url is None:
            server_url = "<unknown>"

        now = time.monotonic()
        # Circuit breaker gate ----------------------------------------------
        if self._breaker_until.get(server_url, 0) > now:
            raise CircuitOpenError(f"Circuit open for server {server_url} - retry later.")

        # Retry policy -----------------------------------------------------
        max_attempts = int(self._retry_policy.get("max_attempts", 1))
        backoff_base = float(self._retry_policy.get("backoff_base", 0.5))

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = await client.call_tool(name=name,
                                                arguments=arguments,
                                                server_args=server_args or {})
                # Success – reset failure counter
                self._fail_log.pop(server_url, None)
                self._breaker_until.pop(server_url, None)
                return result
            except Exception as exc:
                self._fail_log[server_url] = self._fail_log.get(server_url, 0) + 1

                if attempt >= max_attempts:
                    # Trip breaker if threshold met
                    if self._fail_log[server_url] >= self._breaker_threshold:
                        self._breaker_until[server_url] = now + self._breaker_reset_after
                    # Surface original exception so callers/tests can assert properly
                    raise exc

                # Exponential back-off before next attempt
                await asyncio.sleep(delay=backoff_base * (2 ** (attempt - 1)))

        # Fallback – should never hit
        raise ToolRemoteError("Tool call failed after retries (unexpected fall-through).")

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    async def _connect(self, url: str) -> Optional[_Connection]:
        """Try to open an HTTP connection; return ``None`` on failure."""
        client = self._transport_cls(url=url, **self._client_kwargs)
        try:
            await client.__aenter__()
            self._logger.info(msg="Connected to MCP server", 
                              url=url)
            return _Connection(client=client, url=url)
        except Exception as exc:
            self._logger.warning("Failed to connect to MCP server %s: %s", url, exc)
            await client.close()
            return None

    def _find_server_url(self, client: SimpleHTTPTransport) -> Optional[str]:
        return self._client_to_url.get(client)