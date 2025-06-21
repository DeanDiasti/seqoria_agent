from __future__ import annotations

"""FastXMCP FastAPI application."""

from typing import Any, Callable, Dict
import inspect
import itertools
import functools as _ft

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from seqoria_agent.server.formatters import DeclarativeXMLFormatter  # formatter helpers
from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import XMCPError

__all__ = ["FastXMCP"]

# ---------------------------------------------------------------------------
# Registries & helper model
# ---------------------------------------------------------------------------


class _ToolInfo(BaseModel):
    func: Callable[..., Any]
    description: str | None = None
    display: bool = False  # is it a display tool?
    # List of parameter names that should be populated by the server, not the LLM/client.
    server_params: list[str] = []
    # NEW: Explicit tool dependencies (names) for DAG execution
    depends_on: list[str] = []
    # JSON schema describing AI-visible parameters
    input_schema: dict[str, Any] = {}


    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# FastXMCP main application class
# ---------------------------------------------------------------------------


class FastXMCP(FastAPI):
    """A minimal FastAPI app (branded as FastXMCP) that speaks the MCP Streamable-HTTP dialect."""

    def __init__(
        self,
        *,
        title: str = "FastXMCP",
        version: str = "0.1.0",
        strict_mode: bool = False,
        strict_server_params: bool = True,
        **kw,
    ):
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
        self._strict_server_params = bool(strict_server_params)

        # Instance-specific tool registry ---------------------------------
        self._tools: Dict[str, _ToolInfo] = {}

        # Map tool_name → optional custom XML formatter
        self._tool_formatters: dict[str, Any] = {}

        # Registry of server-side parameter provider callables.
        # Each provider must accept (request: Request, tool_name: str, param_name: str)
        # and return the injected value (sync or async). First non-None wins.
        self._param_providers: list[Callable] = []

        # --------------------------------------------------------------
        # Helper: Python → JSON-schema type mapping
        # --------------------------------------------------------------

        def _json_schema_for_type(tp):  # noqa: ANN001
            from typing import get_origin, get_args

            origin = get_origin(tp) or tp
            mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
            }
            if origin in mapping:
                return {"type": mapping[origin]}
            # Fallback catch-all
            return {"type": "string"}

        def _make_reg(display: bool):
            def _decorator(
                func: Callable | None = None,
                *,
                xml_formatter: Any | None = None,
                description: str | None = None,
                server_populated: list[str] | None = None,
                depends_on: list[str] | None = None,
            ):  # noqa: ANN401
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

                    # Build JSON schema for AI-visible parameters -------------------
                    import inspect as _insp

                    sig = _insp.signature(f)
                    props: dict[str, Any] = {}
                    required: list[str] = []

                    for pname, param in sig.parameters.items():
                        if pname in (server_populated or []):
                            # hidden from AI
                            continue

                        annotation = param.annotation if param.annotation is not _insp._empty else str
                        props[pname] = _json_schema_for_type(annotation)

                        if param.default is _insp._empty:
                            required.append(pname)

                    input_schema = {
                        "type": "object",
                        "properties": props,
                    }
                    if required:
                        input_schema["required"] = required

                    # Store registry entry with *wrapped* callable
                    self._tools[f.__name__] = _ToolInfo(
                        func=wrapped,
                        description=description,
                        display=display,
                        server_params=list(server_populated or []),
                        depends_on=list(depends_on or []),
                        input_schema=input_schema,
                    )

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

            rpc_res = await self._dispatch(rpc=rpc_req, request=req)
            return JSONResponse(content=rpc_res.model_dump())

    # ------------------------------------------------------------------
    async def _dispatch(self, rpc: "_RPCReq", request: Request) -> "_RPCRes":  # noqa: C901 – compact
        try:
            match rpc.method:
                case "initialize":
                    return _RPCRes(id=rpc.id, result={"serverName": self.title, "serverVersion": self.version})

                case "tools/list":
                    items = []
                    for name, info in self._tools.items():
                        ann: dict[str, Any] = {
                            "is_display_tool": info.display,
                        }
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
                    tool_name: str = params.get("name")
                    if tool_name not in self._tools:
                        raise XMCPError(code="UNKNOWN_TOOL", detail=tool_name, retryable=False)
                    arguments: dict = params.get("arguments", {})
                    server_args: dict = params.get("serverArgs", {})

                    # --- Validate and merge serverArgs -------------------
                    if server_args:
                        for k in server_args:
                            if k not in self._tools[tool_name].server_params:
                                raise XMCPError(code="UNKNOWN_SERVER_PARAM", detail=k, retryable=False)

                            if k in arguments and self._strict_server_params:
                                raise XMCPError(code="FORBIDDEN_PARAM_OVERRIDE", detail=k, retryable=False)

                        # Merge – values from serverArgs win over arguments (they shouldn't overlap unless override allowed)
                        arguments = {**arguments, **server_args}

                    # --- Resolve any remaining SPP via providers ---------
                    if self._tools[tool_name].server_params:
                        arguments = await self._inject_server_params(
                            request=request,
                            tool_name=tool_name,
                            info=self._tools[tool_name],
                            arguments=arguments,
                        )
                    result = await _maybe_await(self._tools[tool_name].func(**arguments))

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
    # ---  Wrapped tool helpers (ported from legacy qutie)  --------------
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None = None):  # noqa: ANN001
        """Return a wrapper that adds XML wrapping & structured error handling."""

        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls

        self._tool_formatters[tool_name] = formatter_instance

        from seqoria_agent.models.envelope import DisplayEnvelope  # local import to avoid cycle

        async def _async_impl(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
            try:
                result = await func(*args, **kwargs)  # type: ignore[arg-type]
                if self._strict_mode:
                    self._validate_payload(value=result)
                return self._serialize_success(tool_name=tool_name, value=result, formatter=formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name=tool_name, exc=exc)

        def _sync_impl(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = func(*args, **kwargs)
                if self._strict_mode:
                    self._validate_payload(value=result)
                return self._serialize_success(tool_name=tool_name, value=result, formatter=formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name=tool_name, exc=exc)

        wrapper = _async_impl if inspect.iscoroutinefunction(obj=func) else _sync_impl
        _ft.update_wrapper(wrapper=wrapper, wrapped=func)  # type: ignore[arg-type]
        return wrapper

    # ------------------------------------------------------------------
    def _validate_payload(self, value: Any):  # noqa: D401, ANN001
        """Strict-mode payload validation copied from legacy qutie."""

        from seqoria_agent.models.envelope import DisplayEnvelope  # local import

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
        from seqoria_agent.models.envelope import DisplayEnvelope  # local
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
    # NEW – Server-populated parameter injection helper ----------------
    # ------------------------------------------------------------------

    async def _inject_server_params(
        self,
        *,
        request: Request,
        tool_name: str,
        info: _ToolInfo,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:  # noqa: D401
        """Return a new *arguments* dict with server parameters injected.

        The function observes the ``strict_server_params`` flag:
        if the LLM/client tries to supply a value for a protected field and
        *strict* is True, we raise ``XMCPError``. Otherwise, we overwrite.
        """

        injected: dict[str, Any] = dict(arguments)  # shallow copy

        for param in info.server_params:
            # Skip if already supplied (via serverArgs merge)
            if param in injected:
                continue

            value_found = False
            for prov in self._param_providers:
                try:
                    res = prov(request, tool_name, param)
                    if inspect.isawaitable(res):  # type: ignore[arg-type]
                        res = await res  # type: ignore[assignment]
                except Exception:  # pragma: no cover – provider bug
                    continue  # Skip faulty provider

                if res is not None:
                    injected[param] = res
                    value_found = True
                    break

            if not value_found:
                raise XMCPError(code="MISSING_SERVER_PARAM", detail=param, retryable=False)

        return injected

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

    # ------------------------------------------------------------------
    # Public API: parameter provider registration ----------------------
    # ------------------------------------------------------------------

    def register_param_provider(self, fn: Callable):  # noqa: D401
        """Register *fn* as a server parameter provider.

        The function may be synchronous or asynchronous and must accept
        ``(request, tool_name, param_name)``.  The first provider that returns
        a non-None value wins.
        """

        self._param_providers.append(fn)
        return fn


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