from __future__ import annotations

"""Lightweight HTTP transport for seqoria_agent/FastMCP servers.

This wrapper speaks the *streamable-http* JSON-RPC dialect directly via
``httpx`` and provides only the two calls the higher-level ``AIClient``
needs today:  ``tools/list`` and ``tools/call``.

It intentionally does **not** try to implement the full MCP spec - we can
expand it incrementally as our client grows.
"""

from types import SimpleNamespace
from typing import Any, List, Sequence, Optional
import itertools
import httpx
import asyncio

__all__: list[str] = ["SimpleHTTPTransport"]


class SimpleHTTPTransport:
    """Minimal async transport for FastMCP *streamable-http* endpoints."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        headers: Optional[dict[str, str]] = None,
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

    # ------------------------------------------------------------------
    # Async context manager helpers
    # ------------------------------------------------------------------
    async def __aenter__(self):
        # Lazily construct the httpx client so timeout & headers apply
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
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

        msg = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
            "params": params or {},
        }
        resp = await self._client.post(url=self._url, json=msg, follow_redirects=True)
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
                setattr(t, "inputSchema", getattr(t, "parameters"))
            setattr(t, "annotations", getattr(t, "annotations", None))
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