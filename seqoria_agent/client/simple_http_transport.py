from __future__ import annotations

"""Lightweight HTTP transport for seqoria_agent/FastMCP servers.

This wrapper speaks the *streamable-http* JSON-RPC dialect directly via
``httpx`` and provides only the two calls the higher-level ``AIClient``
needs today:  ``tools/list`` and ``tools/call``.

It intentionally does **not** try to implement the full MCP spec - we can
expand it incrementally as our client grows.
"""

import itertools
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence
import inspect

import httpx

from seqoria_agent.auth.providers import BaseAuthProvider

__all__: list[str] = ["SimpleHTTPTransport"]


class SimpleHTTPTransport:
    """Minimal async transport for FastMCP *streamable-http* endpoints."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        headers: Optional[dict[str, str]] = None,
        auth_provider: "BaseAuthProvider" | None = None,
        app: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        # Normalise URL so Starlette's trailing-slash redirect (307 → /mcp/)
        # is avoided entirely.  We *always* keep a trailing slash.
        self._url = url.rstrip("/") + "/"
        self._timeout = timeout
        self._headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **(headers or {}),
        }
        self._client: httpx.AsyncClient | None = None
        self._id_counter = itertools.count(1)

        # Phase-2: dynamic auth provider & in-process testing support ------
        # We keep *references* so that swapping the provider at runtime is
        # picked up by subsequent requests without reconstructing the
        # transport.
        self._auth_provider = auth_provider

        # ``app``/``base_url`` support allows in-memory testing via
        # ``httpx.AsyncClient(app=app, base_url=...)`` which avoids the
        # need for a live HTTP server in the unit-test suite.
        self._asgi_app = app
        self._base_url_override = base_url

    # ------------------------------------------------------------------
    # Async context manager helpers
    # ------------------------------------------------------------------
    async def __aenter__(self):
        client_kwargs: dict[str, Any] = {
            "timeout": self._timeout,
            "headers": self._headers,
        }

        # ------------------------------------------------------------------
        # In-process testing: use ASGITransport for broader version support
        # ------------------------------------------------------------------
        if self._asgi_app is not None:
            try:
                from httpx import ASGITransport  # type: ignore

                transport = ASGITransport(app=self._asgi_app)
                client_kwargs["transport"] = transport
                client_kwargs["base_url"] = self._base_url_override or self._url
            except Exception:  # pragma: no cover – fall back to 'app' param
                client_kwargs["app"] = self._asgi_app  # older httpx versions
                client_kwargs["base_url"] = self._base_url_override or self._url

        self._client = httpx.AsyncClient(**client_kwargs)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):  # noqa: D401 – simple wrapper
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------
    async def _rpc(self, method: str, params: dict | None = None) -> Any:
        if self._client is None:
            raise RuntimeError("Transport not initialised – use 'async with'.")

        # Build request message -----------------------------------------
        msg = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
            "params": params or {},
        }

        # Merge dynamic auth headers – we resolve them *per request* so
        # providers can refresh tokens lazily (e.g., OAuth).  Providers may
        # return a coroutine or plain dict.
        extra_headers: dict[str, str] = {}
        if self._auth_provider is not None:
            # Phase-4: allow provider to refresh tokens proactively
            if hasattr(self._auth_provider, "refresh_token_if_needed"):
                _maybe = self._auth_provider.refresh_token_if_needed()
                if inspect.isawaitable(_maybe):
                    await _maybe

            hdrs = self._auth_provider.auth_headers()
            if inspect.isawaitable(hdrs):  # type: ignore[arg-type]
                hdrs = await hdrs  # type: ignore[assignment]
            if hdrs:
                extra_headers = dict(hdrs)  # defensive copy

        resp = await self._client.post(
            url=self._url,
            json=msg,
            follow_redirects=True,
            headers=extra_headers or None,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body and body["error"] is not None:
            raise RuntimeError(body["error"])
        return body.get("result")

    # ------------------------------------------------------------------
    # Public API – mirrors subset of MCP SDK transport
    # ------------------------------------------------------------------
    async def list_tools(self) -> Sequence[Any]:
        """Return raw tool dicts as provided by the server."""
        result = await self._rpc(method="tools/list")
        # FastMCP paginates; unwrap "items" if present
        if isinstance(result, dict) and "items" in result:
            result = result["items"]
        tools: list[Any] = []
        for t in result or []:
            # Ensure attribute-style access expected by AIClient
            if not isinstance(t, SimpleNamespace):
                t = SimpleNamespace(**t)
            # Older servers may use "parameters" instead of "inputSchema"
            if not hasattr(t, "inputSchema") and hasattr(t, "parameters"):
                t.inputSchema = t.parameters
            t.annotations = getattr(t, "annotations", None)
            tools.append(t)
        return tools

    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any] | None = None,
        server_args: dict[str, Any] | None = None,
    ) -> List[Any]:
        params = {
            "name": name,
            "arguments": arguments or {},
        }
        if server_args:
            params["serverArgs"] = server_args
        result = await self._rpc(method="tools/call", params=params)
        # ``result`` is usually a dict {"contents": [...]} according to MCP
        contents = result
        if isinstance(result, dict) and "contents" in result:
            contents = result["contents"]
        # Wrap each item so ``.text`` access works for text contents
        wrapped: list[Any] = []
        for itm in contents or []:
            if isinstance(itm, dict) and itm.get("type") == "text":
                wrapped.append(SimpleNamespace(text=itm.get("text", "")))
            else:
                wrapped.append(SimpleNamespace(text=str(object=itm)))
        return wrapped

    # ------------------------------------------------------------------
    # Public helper – swap auth provider at runtime
    # ------------------------------------------------------------------

    def set_auth_provider(self, provider) -> None:  # noqa: D401, ANN001
        """Replace the active *provider* instance used for header injection."""
        self._auth_provider = provider 