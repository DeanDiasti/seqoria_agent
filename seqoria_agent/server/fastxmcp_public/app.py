from __future__ import annotations

"""Thin FastXMCP façade (public import location)."""

# The whole original implementation is copied verbatim with the only change
# being the import path used inside `include_router()`.

from typing import Any, Dict, List
import inspect

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from seqoria_agent.server.xmcp_core import AuthBridge, ToolCore
from seqoria_agent.server.xmcp_core.models import ToolInfo as _ToolInfo  # re-export for legacy callers
from seqoria_agent.auth.backends import (
    BaseAuthBackend,
    BaseMiddlewareAuthBackend,
    NoAuthBackend,
)
from seqoria_agent.models.errors import XMCPError

__all__: list[str] = ["FastXMCP"]

# ---------------------------------------------------------------------------
# Tiny JSON-RPC helpers (subset adequate for the test-suite)
# ---------------------------------------------------------------------------


class _RPCReq(BaseModel):
    jsonrpc: str
    id: int | str
    method: str
    params: Dict[str, Any] | None = None


class _RPCRes(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str
    result: Any | None = None
    error: Any | None = None


async def _maybe_await(val):  # noqa: D401 – helper
    if inspect.isawaitable(val):
        return await val  # type: ignore[return-value]
    return val


# ---------------------------------------------------------------------------
# Public façade – trimmed to the essentials
# ---------------------------------------------------------------------------


class FastXMCP(FastAPI):  # noqa: D101 – public server class
    def __init__(
        self,
        *,
        title: str = "FastXMCP",
        name: str | None = None,  # legacy alias used in tests
        version: str = "0.1.0",
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        auth_backend: BaseAuthBackend | None = None,
        **kw,
    ) -> None:
        # Convert ``name=…`` → ``title`` for backwards compatibility.
        if name and title == "FastXMCP":
            title = name
        super().__init__(title=title, version=version, **kw)

        # ------------------------------------------------------------------
        # Core engine (+ public decorators)
        # ------------------------------------------------------------------
        self._core = ToolCore(
            strict_mode=strict_mode,
            strict_server_params=strict_server_params,
            json_wrapper=json_wrapper,
        )
        self.tool = self._core.tool  # type: ignore[attr-defined]
        self.display_tool = self._core.display_tool  # type: ignore[attr-defined]

        # Provide legacy-visible registries
        self._tools = self._core.tools  # type: ignore[attr-defined]
        self._tool_formatters = self._core._tool_formatters  # noqa: SLF001
        self._param_providers = self._core._param_providers  # noqa: SLF001

        # ------------------------------------------------------------------
        # Authentication
        # ------------------------------------------------------------------
        if auth_backend is None:
            auth_backend = NoAuthBackend()
        if not isinstance(auth_backend, BaseAuthBackend):  # pragma: no cover
            raise TypeError("auth_backend must derive from BaseAuthBackend")
        self._auth_backend = auth_backend

        # Auto-mount middleware-capable backends.
        if isinstance(auth_backend, BaseMiddlewareAuthBackend):
            self.add_middleware(type(auth_backend), **{
                k: v for k, v in auth_backend.__dict__.items() if not k.startswith("_")
            })

        self._auth_bridge = AuthBridge(self._auth_backend)
        self._tool_auth_backends: dict[str, BaseAuthBackend] = {}

        # ------------------------------------------------------------------
        # Single JSON-RPC endpoint (/mcp/)
        # ------------------------------------------------------------------
        @self.post("/mcp/", include_in_schema=False)
        async def _mcp(req: Request):  # noqa: D401 – endpoint closure
            try:
                rpc_in = _RPCReq.model_validate(await req.json())
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({"error": str(exc)}, status_code=400)
            rpc_out = await self._dispatch(rpc_in, req)
            return JSONResponse(rpc_out.model_dump())

    # ------------------------------------------------------------------
    async def _dispatch(self, rpc: _RPCReq, request: Request) -> _RPCRes:  # noqa: C901 – concise
        try:
            match rpc.method:
                case "initialize":
                    return _RPCRes(id=rpc.id, result={"serverName": self.title, "serverVersion": self.version})

                case "tools/list":
                    items: List[Dict[str, Any]] = []
                    for name, info in self._tools.items():
                        ann = {"is_display_tool": info.display}
                        if info.server_params:
                            ann["server_populated"] = info.server_params
                        if info.depends_on:
                            ann["depends_on"] = info.depends_on
                        items.append(
                            {
                                "name": name,
                                "description": info.description,
                                "inputSchema": info.input_schema,
                                "annotations": ann,
                            }
                        )
                    return _RPCRes(id=rpc.id, result={"items": items})

                case "tools/call":
                    params = rpc.params or {}
                    tool_name: str = params.get("name")  # type: ignore[assignment]
                    if tool_name not in self._tools:
                        raise XMCPError(code="UNKNOWN_TOOL", detail=tool_name, retryable=False)

                    backend = self._tool_auth_backends.get(tool_name, self._auth_backend)
                    bridge = self._auth_bridge if backend is self._auth_backend else AuthBridge(backend)
                    await bridge.authenticate(request)
                    bridge.ensure_principal_or_error(tool_name)

                    arguments: dict = params.get("arguments", {}) or {}
                    server_args: dict = params.get("serverArgs", {}) or {}

                    if self._tools[tool_name].server_params or server_args:
                        arguments = await self._core.inject_server_params(
                            request_or_ns=request,
                            tool_name=tool_name,
                            info=self._tools[tool_name],
                            arguments=arguments,
                            server_args=server_args,
                        )
                    xml = await self._core.run_tool(tool_name, arguments)
                    return _RPCRes(id=rpc.id, result={"contents": [{"type": "text", "text": xml}]})

                case _:
                    raise XMCPError(code="METHOD_NOT_FOUND", detail=rpc.method, retryable=False)
        except XMCPError as xe:
            return _RPCRes(id=rpc.id, error={"code": xe.code, "message": xe.detail})
        except Exception as exc:  # noqa: BLE001 – propagate HTTPException
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                raise
            return _RPCRes(id=rpc.id, error={"code": "SERVER_ERROR", "message": str(exc)})

    # ------------------------------------------------------------------
    # Public helpers / shims
    # ------------------------------------------------------------------

    def register_param_provider(self, fn):  # noqa: D401
        return self._core.register_param_provider(fn)

    def include_router(self, router, **kw):  # noqa: D401 – merge registries
        super().include_router(router, **kw)
        try:
            from seqoria_agent.server.fastxmcp_public.router import FastXMCPRouter  # updated import
        except Exception:  # pragma: no cover – missing module
            FastXMCPRouter = None  # type: ignore
        if FastXMCPRouter is not None and isinstance(router, FastXMCPRouter):  # type: ignore[arg-type]
            for name, info in router._tools.items():  # type: ignore[attr-defined]
                if name in self._tools:
                    raise XMCPError(code="DUPLICATE_TOOL", detail=name, retryable=False)
                self._tools[name] = info
                if name in router._tool_formatters:  # type: ignore[attr-defined]
                    self._tool_formatters[name] = router._tool_formatters[name]  # type: ignore[index]
                self._tool_auth_backends[name] = router._auth_backend  # type: ignore[attr-defined]
            self._param_providers.extend(router._param_providers)  # type: ignore[attr-defined]
        return router

    # Tests import this utility directly → delegate to ToolCore
    def _wrap_tool_with_error_handling(self, func, tool_name: str, formatter_cls: Any | None = None):  # noqa: D401, ANN001
        return self._core._wrap_tool_with_error_handling(func, tool_name, formatter_cls) 