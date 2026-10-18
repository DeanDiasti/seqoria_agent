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

OAuth 2.1 Support:
- Automatic OAuth flow for protected MCP servers
- Browser-based authorization with PKCE
- Token storage and refresh handling
"""

from __future__ import annotations

import asyncio
import hashlib
import shlex
import webbrowser
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Dict, Literal, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

# OAuth support from MCP SDK
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

__all__ = [
    "MCPTransportFactory",
    "MCPServerConfig",
    "MCPConnection",
    "TransportType",
    "HTTPClientPool",
    "InMemoryTokenStorage",
    "OAuthConfig",
]

TransportType = Literal["http", "sse", "stdio", "auto"]


# ---------------------------------------------------------------------------
# OAuth Token Storage
# ---------------------------------------------------------------------------


class InMemoryTokenStorage(TokenStorage):
    """In-memory token storage implementation for OAuth.

    Stores OAuth tokens and client registration info in memory.
    Tokens are lost when the process exits.

    For persistent storage, subclass TokenStorage and implement
    file-based or database-backed storage.
    """

    def __init__(self) -> None:
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self.client_info = client_info


# ---------------------------------------------------------------------------
# OAuth Configuration
# ---------------------------------------------------------------------------


@dataclass
class OAuthConfig:
    """Configuration for OAuth 2.1 authentication.

    Attributes
    ----------
    client_name : str
        Human-readable name for the OAuth client (shown during consent).
    redirect_uri : str
        URI where the authorization server redirects after consent.
        Default is localhost for CLI/desktop apps.
    scopes : list[str] | None
        OAuth scopes to request. If None, uses server defaults.
    storage : TokenStorage | None
        Token storage implementation. Defaults to InMemoryTokenStorage.
    redirect_handler : callable | None
        Custom handler for opening the auth URL. Defaults to webbrowser.open.
    callback_handler : callable | None
        Custom handler for receiving the callback. Defaults to CLI prompt.
    """

    client_name: str = "Seqoria Agent"
    # When set to "auto" (default), seqoria-agent starts a local loopback HTTP server
    # on 127.0.0.1 with a random free port and automatically captures the OAuth
    # callback without requiring you to run a separate webserver.
    #
    # This avoids browser errors like ERR_CONNECTION_REFUSED that happen when the
    # redirect URI points at localhost but nothing is listening.
    redirect_uri: str = "auto"
    scopes: list[str] | None = None
    storage: TokenStorage | None = None
    redirect_handler: Callable[[str], Awaitable[None]] | None = None
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None


# ---------------------------------------------------------------------------
# Default OAuth Handlers
# ---------------------------------------------------------------------------


async def _default_redirect_handler(auth_url: str) -> None:
    """Default handler: open auth URL in browser and print instructions."""
    print()
    print("=" * 60)
    print("OAuth Authorization Required")
    print("=" * 60)
    print()
    print("Opening browser for authentication...")
    print()
    print("If the browser doesn't open, visit this URL:")
    print(f"  {auth_url}")
    print()

    # Try to open browser
    try:
        webbrowser.open(auth_url)
    except Exception:
        print("(Could not open browser automatically)")


async def _default_callback_handler() -> tuple[str, str | None]:
    """Default handler: prompt user to paste the callback URL."""
    print("After authorizing, you'll be redirected to a localhost URL.")
    print("Copy the FULL URL from your browser's address bar and paste it below.")
    print()

    callback_url = input("Paste callback URL: ").strip()

    # Parse the callback URL to extract code and state
    parsed = urlparse(callback_url)
    params = parse_qs(parsed.query)

    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]

    if not code:
        raise ValueError("No authorization code found in callback URL")

    print()
    print("Authorization successful!")
    print("=" * 60)
    print()

    return code, state


async def _start_loopback_callback_server(
    *,
    path: str = "/callback",
    host: str = "127.0.0.1",
    timeout_s: float = 300.0,
) -> tuple[str, Callable[[], Awaitable[tuple[str, str | None]]]]:
    """Start a tiny loopback HTTP server to capture OAuth callback parameters.

    Returns (redirect_uri, callback_handler).
    """
    loop = asyncio.get_running_loop()
    result_fut: asyncio.Future[tuple[str, str | None]] = loop.create_future()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.read(8192)
            # Parse request line: GET /callback?code=...&state=... HTTP/1.1
            line = data.split(b"\r\n", 1)[0].decode("utf-8", errors="ignore")
            parts = line.split()
            req_target = parts[1] if len(parts) >= 2 else "/"

            parsed = urlparse(req_target)
            qs = parse_qs(parsed.query)
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]

            if code and not result_fut.done():
                result_fut.set_result((code, state))

            body = (
                "<html><body>"
                "<h2>Authorization complete</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
                "</body></html>"
            ).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("utf-8")
                + b"Connection: close\r\n"
                b"\r\n" + body
            )
            await writer.drain()
        except Exception:
            # Best-effort: do not crash the auth flow on callback handling issues.
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(_handle, host=host, port=0)
    sock = server.sockets[0] if server.sockets else None
    port = sock.getsockname()[1] if sock is not None else 0
    redirect_uri = f"http://{host}:{port}{path}"

    async def _callback_handler() -> tuple[str, str | None]:
        try:
            return await asyncio.wait_for(result_fut, timeout=timeout_s)
        finally:
            server.close()
            with suppress(Exception):
                await server.wait_closed()

    return redirect_uri, _callback_handler


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

        pool_key = f"{self._get_pool_key(url, headers)}|t={timeout or self._default_timeout}"

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
    oauth : OAuthConfig | bool | None
        OAuth configuration for protected servers.
        - None or False: No OAuth (use headers for auth if needed)
        - True: Enable OAuth with default settings
        - OAuthConfig: Enable OAuth with custom settings
    """

    url: str
    transport: TransportType = "auto"
    headers: dict[str, str] | None = None
    timeout: float = 30.0
    env: dict[str, str] | None = None
    oauth: OAuthConfig | bool | None = None


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
        elif url_lower.startswith(("sse://", "sses://")):
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
        parsed = urlparse(url)
        if parsed.path in ("", "/"):
            url = parsed._replace(path="/mcp").geturl()

        # Check if OAuth is enabled
        if config.oauth:
            async with MCPTransportFactory._connect_http_oauth(config, url) as conn:
                yield conn
            return

        # Non-OAuth path: use shared client pool for connection reuse
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
    async def _connect_http_oauth(
        config: MCPServerConfig,
        url: str,
    ) -> AsyncIterator[MCPConnection]:
        """Connect via Streamable HTTP transport with OAuth 2.1 authentication.

        This implements the full MCP OAuth flow:
        1. Protected Resource Metadata discovery
        2. Authorization Server discovery
        3. Dynamic Client Registration (or use existing)
        4. Browser-based authorization with PKCE
        5. Token exchange and storage

        Parameters
        ----------
        config : MCPServerConfig
            Server configuration with OAuth enabled.
        url : str
            Normalized MCP endpoint URL.
        """
        # Resolve OAuth config
        if config.oauth is True:
            oauth_cfg = OAuthConfig()
        elif isinstance(config.oauth, OAuthConfig):
            oauth_cfg = config.oauth
        else:
            raise ValueError("OAuth config must be True or OAuthConfig instance")

        # Get server base URL (without /mcp path)
        parsed = urlparse(url)
        server_url = f"{parsed.scheme}://{parsed.netloc}"

        # Set up token storage
        storage = oauth_cfg.storage or InMemoryTokenStorage()

        # Set up handlers
        redirect_handler = oauth_cfg.redirect_handler or _default_redirect_handler

        # If redirect_uri is "auto", start a loopback server that captures the callback.
        # This avoids requiring a user-operated copy/paste and prevents browser
        # connection-refused errors on localhost.
        callback_handler = oauth_cfg.callback_handler
        redirect_uri = oauth_cfg.redirect_uri
        if redirect_uri in ("auto", "", None):
            redirect_uri, callback_handler = await _start_loopback_callback_server()
        if callback_handler is None:
            callback_handler = _default_callback_handler

        # Build redirect URIs list
        redirect_uris = [AnyUrl(redirect_uri)]

        # Build client metadata
        client_metadata = OAuthClientMetadata(
            client_name=oauth_cfg.client_name,
            redirect_uris=redirect_uris,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=" ".join(oauth_cfg.scopes) if oauth_cfg.scopes else None,
        )

        # Create OAuth provider
        oauth_auth = OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

        # Create HTTP client with OAuth auth
        timeout = httpx.Timeout(config.timeout, read=300.0)
        async with httpx.AsyncClient(
            auth=oauth_auth,
            follow_redirects=True,
            timeout=timeout,
            http2=True,
        ) as http_client:
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

        async with sse_client(url, headers=config.headers, timeout=config.timeout) as (read, write):
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
