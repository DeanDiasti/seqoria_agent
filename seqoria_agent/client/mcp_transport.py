"""MCP Transport Factory for unified transport management.

This module provides a factory for creating MCP SDK transports with support for:
- Streamable HTTP (http://, https://)
- SSE (sse://)
- stdio (stdio://)

Auto-detection based on URL scheme or explicit configuration.

Performance Features:
- Shared httpx.AsyncClient pool for HTTP connection reuse
- Automatic keep-alive management
- Configurable connection limits
"""

from __future__ import annotations

import asyncio
import hashlib
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar, Dict, Literal, Optional
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

__all__ = [
    "MCPTransportFactory",
    "MCPServerConfig",
    "MCPConnection",
    "TransportType",
    "HTTPClientPool",
]

TransportType = Literal["http", "sse", "stdio", "auto"]


class HTTPClientPool:
    """Shared pool of httpx.AsyncClient instances for connection reuse.
    
    This pool maintains one AsyncClient per unique (host, port) combination,
    enabling HTTP connection reuse (keep-alive) across multiple MCP requests.
    
    Thread-safe via asyncio.Lock.
    
    Example
    -------
    >>> pool = HTTPClientPool()
    >>> async with pool.get_client("https://api.example.com/mcp") as client:
    ...     # Uses shared client with connection pooling
    ...     response = await client.get("/health")
    >>> await pool.close()  # Cleanup when done
    """
    
    _instance: ClassVar[Optional["HTTPClientPool"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    
    def __init__(
        self,
        *,
        default_timeout: float = 30.0,
        read_timeout: float = 300.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        """Initialize the HTTP client pool.
        
        Parameters
        ----------
        default_timeout : float
            Default timeout for requests in seconds.
        read_timeout : float
            Read timeout for streaming responses (SSE) in seconds.
        max_connections : int
            Maximum number of total connections in the pool.
        max_keepalive_connections : int
            Maximum number of keep-alive connections per host.
        """
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_lock = asyncio.Lock()
        self._default_timeout = default_timeout
        self._read_timeout = read_timeout
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._closed = False
    
    @classmethod
    async def get_shared(cls) -> "HTTPClientPool":
        """Get or create the shared singleton instance."""
        async with cls._lock:
            if cls._instance is None or cls._instance._closed:
                cls._instance = cls()
            return cls._instance
    
    @classmethod
    async def close_shared(cls) -> None:
        """Close the shared singleton instance."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None
    
    def _headers_fingerprint(self, headers: Dict[str, str] | None) -> str:
        """Return a stable fingerprint for headers without retaining raw values.

        This prevents accidentally reusing an httpx.AsyncClient configured with
        one auth header set for requests that need a different header set.
        """
        if not headers:
            return "nohdr"

        h = hashlib.sha1()
        for k, v in sorted(headers.items()):
            h.update(str(k).lower().encode("utf-8"))
            h.update(b"\0")
            h.update(str(v).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _get_pool_key(self, url: str, headers: Dict[str, str] | None) -> str:
        """Generate pool key from URL (scheme://host:port) + header fingerprint."""
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        hdr_fp = self._headers_fingerprint(headers)
        return f"{parsed.scheme}://{parsed.hostname}:{port}|h={hdr_fp}"
    
    async def get_client(
        self,
        url: str,
        *,
        headers: Dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.AsyncClient:
        """Get or create an AsyncClient for the given URL's host.
        
        Parameters
        ----------
        url : str
            Target URL (used to determine host for pooling).
        headers : dict | None
            Additional headers to include in requests.
        timeout : float | None
            Custom timeout override.
            
        Returns
        -------
        httpx.AsyncClient
            Shared or new client instance.
        """
        if self._closed:
            raise RuntimeError("HTTPClientPool is closed")
        
        pool_key = self._get_pool_key(url, headers)
        
        async with self._client_lock:
            if pool_key not in self._clients:
                effective_timeout = httpx.Timeout(
                    timeout or self._default_timeout,
                    read=self._read_timeout,
                )
                self._clients[pool_key] = httpx.AsyncClient(
                    headers=headers or {},
                    timeout=effective_timeout,
                    limits=self._limits,
                    http2=True,  # Enable HTTP/2 for better multiplexing
                )
            return self._clients[pool_key]
    
    async def close(self) -> None:
        """Close all pooled clients and release resources."""
        async with self._client_lock:
            self._closed = True
            for client in self._clients.values():
                try:
                    await client.aclose()
                except Exception:
                    pass
            self._clients.clear()


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes
    ----------
    url : str
        The server URL or command for stdio transport.
        - For HTTP: "http://localhost:8000/mcp" or "https://..."
        - For SSE: "sse://localhost:8000/events"
        - For stdio: "stdio://python -m my_server" or "stdio:///path/to/server"
    transport : TransportType
        Transport type: "http", "sse", "stdio", or "auto" (default).
        When "auto", the transport is inferred from the URL scheme.
    headers : dict[str, str] | None
        Optional HTTP headers for HTTP/SSE transports.
    timeout : float
        Connection timeout in seconds (default: 30.0).
    env : dict[str, str] | None
        Environment variables for stdio subprocess.
    """

    url: str
    transport: TransportType = "auto"
    headers: dict[str, str] | None = None
    timeout: float = 30.0
    env: dict[str, str] | None = None


@dataclass
class MCPConnection:
    """Wrapper for an active MCP session with its transport context.

    Attributes
    ----------
    session : ClientSession
        The MCP ClientSession for this connection.
    url : str
        The original URL/command used to establish the connection.
    transport_type : TransportType
        The resolved transport type.
    """

    session: ClientSession
    url: str
    transport_type: TransportType
    _cleanup: Any = field(default=None, repr=False)

    async def close(self) -> None:
        """Close the connection and cleanup resources."""
        if self._cleanup is not None:
            try:
                await self._cleanup.__aexit__(None, None, None)
            except Exception:
                pass


class MCPTransportFactory:
    """Factory for creating MCP transport connections.

    Supports automatic transport type detection based on URL scheme:
    - http://, https:// -> Streamable HTTP
    - sse:// -> Server-Sent Events
    - stdio:// -> Standard I/O subprocess

    Example
    -------
    >>> async with MCPTransportFactory.connect("http://localhost:8000/mcp") as conn:
    ...     tools = await conn.session.list_tools()
    """

    @staticmethod
    def detect_transport(url: str) -> TransportType:
        """Detect the transport type from URL scheme.

        Parameters
        ----------
        url : str
            The server URL.

        Returns
        -------
        TransportType
            The detected transport type.
        """
        url_lower = url.lower()
        if url_lower.startswith("http://") or url_lower.startswith("https://"):
            return "http"
        elif url_lower.startswith("sse://"):
            return "sse"
        elif url_lower.startswith("stdio://"):
            return "stdio"
        else:
            # Default to HTTP for URLs without explicit scheme
            return "http"

    @staticmethod
    @asynccontextmanager
    async def connect(
        config: str | MCPServerConfig,
        *,
        client_pool: HTTPClientPool | None = None,
    ) -> AsyncIterator[MCPConnection]:
        """Create an MCP connection with automatic transport detection.

        Parameters
        ----------
        config : str | MCPServerConfig
            Either a URL string or full MCPServerConfig.
        client_pool : HTTPClientPool | None
            Optional shared HTTP client pool for connection reuse.
            If None, uses the global shared pool for HTTP transports.

        Yields
        ------
        MCPConnection
            The active MCP connection.
        """
        # Normalize config
        if isinstance(config, str):
            config = MCPServerConfig(url=config, transport="auto")

        # Resolve transport type
        transport_type = config.transport
        if transport_type == "auto":
            transport_type = MCPTransportFactory.detect_transport(config.url)

        # Create appropriate transport
        if transport_type == "http":
            async with MCPTransportFactory._connect_http(config, client_pool=client_pool) as conn:
                yield conn
        elif transport_type == "sse":
            async with MCPTransportFactory._connect_sse(config) as conn:
                yield conn
        elif transport_type == "stdio":
            async with MCPTransportFactory._connect_stdio(config) as conn:
                yield conn
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

    @staticmethod
    @asynccontextmanager
    async def _connect_http(
        config: MCPServerConfig,
        *,
        client_pool: HTTPClientPool | None = None,
    ) -> AsyncIterator[MCPConnection]:
        """Connect via Streamable HTTP transport.
        
        Parameters
        ----------
        config : MCPServerConfig
            Server configuration.
        client_pool : HTTPClientPool | None
            Optional shared client pool. If None, uses the global shared pool.
        """
        url = config.url
        # Normalize URL to include /mcp if not present
        if not url.endswith("/mcp") and not url.endswith("/mcp/"):
            url = url.rstrip("/") + "/mcp"

        # Use shared client pool for connection reuse
        if client_pool is None:
            client_pool = await HTTPClientPool.get_shared()
        
        http_client = await client_pool.get_client(
            url,
            headers=config.headers,
            timeout=config.timeout,
        )
        
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield MCPConnection(
                    session=session,
                    url=config.url,
                    transport_type="http",
                )

    @staticmethod
    @asynccontextmanager
    async def _connect_sse(config: MCPServerConfig) -> AsyncIterator[MCPConnection]:
        """Connect via SSE transport."""
        # Convert sse:// to http:// for the actual connection
        url = config.url
        if url.startswith("sse://"):
            url = "http://" + url[6:]
        elif url.startswith("sses://"):
            url = "https://" + url[7:]

        # MCP SDK's sse_client takes only the URL
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield MCPConnection(
                    session=session,
                    url=config.url,
                    transport_type="sse",
                )

    @staticmethod
    @asynccontextmanager
    async def _connect_stdio(config: MCPServerConfig) -> AsyncIterator[MCPConnection]:
        """Connect via stdio transport (subprocess)."""
        # Parse stdio:// URL to extract command
        url = config.url
        if url.startswith("stdio://"):
            command_str = url[8:]
        else:
            command_str = url

        # Parse command and arguments
        parts = shlex.split(command_str)
        if not parts:
            raise ValueError(f"Invalid stdio URL: {config.url}")

        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=config.env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield MCPConnection(
                    session=session,
                    url=config.url,
                    transport_type="stdio",
                )


async def connect_mcp(
    url: str,
    *,
    transport: TransportType = "auto",
    client_pool: HTTPClientPool | None = None,
) -> AsyncIterator[MCPConnection]:
    """Convenience function to connect to an MCP server.

    Parameters
    ----------
    url : str
        The server URL.
    transport : TransportType
        Transport type (default: auto-detect).
    client_pool : HTTPClientPool | None
        Optional shared HTTP client pool for connection reuse.

    Returns
    -------
    AsyncIterator[MCPConnection]
        Context manager yielding the MCP connection.

    Example
    -------
    >>> async with connect_mcp("http://localhost:8000/mcp") as conn:
    ...     tools = await conn.session.list_tools()
    """
    config = MCPServerConfig(url=url, transport=transport)
    async with MCPTransportFactory.connect(config, client_pool=client_pool) as conn:
        yield conn
