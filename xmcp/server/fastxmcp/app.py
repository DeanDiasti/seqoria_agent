from __future__ import annotations

"""FastXMCP FastAPI application (migrated).

This module contains the original implementation that used to live in
``xmcp.server.fastxmcp`` so that external behaviour remains unchanged
while we refactor the package structure.
"""

# NOTE: This code is an *as-is* copy of the legacy ``fastxmcp.py`` file to
# ensure functional parity during the initial refactor.

from typing import Any, Callable, Dict
import inspect
import itertools
import functools as _ft

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from xmcp.server.formatters import DeclarativeXMLFormatter  # formatter helpers
from xmcp.models.envelope import DisplayEnvelope
from xmcp.models.errors import XMCPError

__all__ = ["FastXMCP"]

# ---------------------------------------------------------------------------
# Registries & helper model
# ---------------------------------------------------------------------------


class _ToolInfo(BaseModel):
    func: Callable[..., Any]
    description: str | None = None
    display: bool = False  # is it a display tool?


    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# FastXMCP main application class
# ---------------------------------------------------------------------------


class FastXMCP(FastAPI):
    """A minimal FastAPI app that speaks the MCP Streamable-HTTP dialect."""

    def __init__(self, *, title: str = "FastXMCP", version: str = "0.1.0", strict_mode: bool = False, **kw):
        """Create a FastXMCP application.

        Parameters
        ----------
        strict_mode : bool, default False
            When *True* every tool return value is validated (DisplayEnvelope
            or XML). Invalid payloads are transformed into a structured error
            envelope to keep the protocol stable.
        """

        super().__init__(title=title, version=version, **kw)

        self._strict_mode = strict_mode

        # Instance-specific tool registry ---------------------------------
        self._tools: Dict[str, _ToolInfo] = {}

        # Map tool_name → optional custom XML formatter
        self._tool_formatters: dict[str, Any] = {}

        def _make_reg(display: bool):
            def _decorator(func: Callable | None = None, *, xml_formatter: Any | None = None, description: str | None = None):  # noqa: ANN401
                """Decorator factory for @app.tool / @app.display_tool.

                Supports both usages:

                @app.tool
                def fn(...): ...

                @app.tool(xml_formatter=Formatter)
                def fn(...): ...
                """

                def _register(f: Callable):
                    # Wrap with error handling + validation first
                    wrapped = self._wrap_tool_with_error_handling(
                        func=f, tool_name=f.__name__, formatter_cls=xml_formatter
                    )

                    # Store registry entry with *wrapped* callable
                    self._tools[f.__name__] = _ToolInfo(func=wrapped, description=description, display=display)

                    # No FastMCP base here, so just return wrapped; dispatcher calls self._tools mapping
                    # but we still want the original function name for introspection, so keep name.
                    return wrapped

                # Without parentheses
                if func is not None and callable(func):
                    return _register(func)

                # With parentheses
                def decorator(fn: Callable):
                    return _register(fn)

                return decorator

            return _decorator

        # Public decorator factories
        self.tool = _make_reg(display=False)  # type: ignore[attr-defined]
        self.display_tool = _make_reg(display=True)  # type: ignore[attr-defined]

        # -----------------------------------------------------------------
        # JSON-RPC endpoint (single POST) ---------------------------------
        # -----------------------------------------------------------------

        self._msg_counter = itertools.count(1)

        # Single endpoint for JSON-RPC
        @self.post("/mcp/", include_in_schema=False)
        async def _mcp_endpoint(req: Request):  # noqa: D401 – closure
            try:
                raw = await req.json()
                rpc_req = _RPCReq.model_validate(raw)
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

            rpc_res = await self._dispatch(rpc_req)
            return JSONResponse(rpc_res.model_dump())

    # ------------------------------------------------------------------
    async def _dispatch(self, rpc: "_RPCReq") -> "_RPCRes":  # noqa: C901 – compact
        try:
            match rpc.method:
                case "initialize":
                    return _RPCRes(id=rpc.id, result={"serverName": self.title, "serverVersion": self.version})

                case "tools/list":
                    items = [
                        {
                            "name": name,
                            "description": info.description,
                            "inputSchema": {"type": "object", "properties": {}},
                            "annotations": {"is_display_tool": info.display},
                        }
                        for name, info in self._tools.items()
                    ]
                    return _RPCRes(id=rpc.id, result={"items": items})

                case "tools/call":
                    params = rpc.params or {}
                    tool_name: str = params.get("name")
                    arguments: dict = params.get("arguments", {})
                    if tool_name not in self._tools:
                        raise XMCPError(code="UNKNOWN_TOOL", detail=tool_name, retryable=False)
                    info = self._tools[tool_name]
                    result = await _maybe_await(info.func(**arguments))

                    # Minimal XML wrapping (reuse existing formatter)
                    fmt = DeclarativeXMLFormatter()
                    if isinstance(result, DisplayEnvelope):
                        display_json = result.dict_for_transport()
                        xml = (
                            f"<tool_response tool_name=\"{tool_name}\">"
                            f"  <display><![CDATA[{display_json}]]></display>\n"
                            f"</tool_response>"
                        )
                    else:
                        xml_body = fmt(result, indent_level=2)
                        xml = (
                            f"<tool_response tool_name=\"{tool_name}\">"
                            f"  <llm_output><![CDATA[\n{xml_body}\n  ]]></llm_output>\n"
                            f"</tool_response>"
                        )

                    return _RPCRes(id=rpc.id, result={"contents": [{"type": "text", "text": xml}]})

                case _:
                    raise XMCPError(code="METHOD_NOT_FOUND", detail=rpc.method, retryable=False)
        except XMCPError as xe:
            return _RPCRes(id=rpc.id, error={"code": xe.code, "message": xe.detail})
        except Exception as exc:
            return _RPCRes(id=rpc.id, error={"code": "SERVER_ERROR", "message": str(exc)})

    # ------------------------------------------------------------------
    # ---  Wrapped tool helpers (ported from legacy XMCP)  --------------
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None = None):  # noqa: ANN001
        """Return a wrapper that adds XML wrapping & structured error handling."""

        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls

        self._tool_formatters[tool_name] = formatter_instance

        from xmcp.models.envelope import DisplayEnvelope  # local import to avoid cycle

        async def _async_impl(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = await func(*args, **kwargs)  # type: ignore[arg-type]
                if self._strict_mode:
                    self._validate_payload(result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        def _sync_impl(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = func(*args, **kwargs)
                if self._strict_mode:
                    self._validate_payload(result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        wrapper = _async_impl if inspect.iscoroutinefunction(func) else _sync_impl
        _ft.update_wrapper(wrapper, func)  # type: ignore[arg-type]
        return wrapper

    # ------------------------------------------------------------------
    def _validate_payload(self, value: Any):  # noqa: D401, ANN001
        """Strict-mode payload validation copied from legacy XMCP."""

        from xmcp.models.envelope import DisplayEnvelope  # local import

        if isinstance(value, DisplayEnvelope):
            return value

        if isinstance(value, (dict, list)):
            # Try coercing into DisplayEnvelope
            try:
                DisplayEnvelope.model_validate(value)
            except Exception as exc:
                raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail=str(exc))
            return value

        if isinstance(value, str):
            xml_s = value.lstrip()
            if xml_s.startswith("<tool_response"):
                import xml.etree.ElementTree as _ET
                try:
                    _ET.fromstring(xml_s)
                    return value
                except Exception as exc:
                    raise XMCPError(code="INVALID_XML", retryable=False, detail=str(exc))

        raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail="Return value must be DisplayEnvelope or <tool_response> XML")

    # ------------------------------------------------------------------
    @staticmethod
    def _build_error_xml(tool_name: str, exc: XMCPError) -> str:  # noqa: D401
        cddata = (exc.detail or exc.code).strip()
        meta_block = ""
        if exc.meta:
            import json as _json
            try:
                meta_json = _json.dumps(exc.meta, separators=(",", ":"))
                meta_block = f"\n    <meta>{meta_json}</meta>"
            except Exception:
                pass

        return (
            f"<tool_response tool_name=\"{tool_name}\">\n"
            f"  <error code=\"{exc.code}\" retryable=\"{'true' if exc.retryable else 'false'}\">\n"
            f"    <![CDATA[{cddata}]]>" + meta_block + "\n  </error>\n"
            f"</tool_response>"
        )

    # ------------------------------------------------------------------
    def _serialize_success(self, tool_name: str, value: Any, formatter: Any | None):  # noqa: ANN001
        from xmcp.models.envelope import DisplayEnvelope  # local
        import json as _json

        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            return value

        display_block = ""
        llm_output_xml = ""

        if isinstance(value, DisplayEnvelope):
            display_json = _json.dumps(value.dict_for_transport(), separators=(",", ":"))
            display_block = f"\n  <display><![CDATA[{display_json}]]></display>"
        else:
            if formatter is None:
                formatter = DeclarativeXMLFormatter()
            formatted = formatter(value, indent_level=2)
            llm_output_xml = f"\n  <llm_output><![CDATA[\n{formatted}\n  ]]></llm_output>"

        return f"<tool_response tool_name=\"{tool_name}\">{llm_output_xml}{display_block}\n</tool_response>"

    # ------------------------------------------------------------------
    # Convenience runner ------------------------------------------------
    # ------------------------------------------------------------------

    def run(self, *, host: str = "127.0.0.1", port: int = 8000, **uvicorn_kwargs):
        """Start an in-process Uvicorn server.

        This mirrors FastAPI's recommended production entry-point but avoids
        repetitive boilerplate in small example scripts.
        """

        import uvicorn

        uvicorn.run(self, host=host, port=port, **uvicorn_kwargs)


async def _maybe_await(v):  # small helper
    if inspect.isawaitable(v):
        return await v
    return v


# ---------------------------------------------------------------------------
# Minimal JSON-RPC models (subset used by AIClient) – RESTORED
# ---------------------------------------------------------------------------


class _RPCReq(BaseModel):
    jsonrpc: str
    id: int | str
    method: str
    params: dict[str, Any] | None = None


class _RPCRes(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str
    result: Any | None = None
    error: Any | None = None 