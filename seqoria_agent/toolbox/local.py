from __future__ import annotations

"""Pure-Python toolbox for defining XMCP-style tools inline.

The implementation mirrors :class:`FastXMCPRouter` but without any FastAPI or
network dependencies.  Functions registered via ``@tool`` / ``@display_tool``
are executed in-process.  The wrapper still serialises results into the
standard ``<tool_response>`` XML so :class:`AIClient` can parse them without
special-casing.
"""

import functools as _ft
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.fastxmcp.app import _ToolInfo  # reuse same helper model
from seqoria_agent.server.formatters import DeclarativeXMLFormatter, derive_formatter_from_data
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY  # central constant
from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend

__all__: list[str] = ["LocalToolbox"]


class LocalToolbox:  # noqa: D401 – simple container
    """Registry for inline XMCP tools (no HTTP layer)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        auth_backend: BaseAuthBackend | None = None,
    ) -> None:
        self._strict_mode = bool(strict_mode)
        self._strict_server_params = bool(strict_server_params)
        self._json_wrapper = bool(json_wrapper)

        if auth_backend is None:
            auth_backend = NoAuthBackend()
        if not isinstance(auth_backend, BaseAuthBackend):
            raise TypeError("auth_backend must be an instance of BaseAuthBackend")

        self._auth_backend: BaseAuthBackend = auth_backend

        self._tools: Dict[str, _ToolInfo] = {}
        self._tool_formatters: dict[str, Any] = {}
        self._param_providers: list[Callable] = []

        # --------------------------------------------------------------
        # Decorator factory (identical to FastXMCPRouter)
        # --------------------------------------------------------------

        def _json_schema_for_type(tp):  # noqa: ANN001
            from typing import get_origin

            origin = get_origin(tp) or tp
            mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
            if origin in mapping:
                return {"type": mapping[origin]}
            return {"type": "string"}

        def _make_reg(display: bool):
            def _decorator(
                func: Callable | None = None,
                *,
                xml_formatter: Any | None = None,
                response_format: Any | None = None,
                description: str | None = None,
                server_populated: list[str] | None = None,
                depends_on: list[str] | None = None,
            ):
                """Register a function as a tool/display_tool."""

                def _register(f: Callable):
                    formatter_cls = response_format if response_format is not None else xml_formatter
                    wrapped = self._wrap_tool_with_error_handling(
                        func=f, tool_name=f.__name__, formatter_cls=formatter_cls
                    )

                    sig = inspect.signature(f)
                    props: dict[str, Any] = {}
                    required: list[str] = []
                    for pname, param in sig.parameters.items():
                        if pname in (server_populated or []):
                            continue
                        annotation = param.annotation if param.annotation is not inspect._empty else str
                        props[pname] = _json_schema_for_type(annotation)
                        if param.default is inspect._empty:
                            required.append(pname)

                    input_schema = {"type": "object", "properties": props}
                    if required:
                        input_schema["required"] = required

                    self._tools[f.__name__] = _ToolInfo(
                        func=wrapped,
                        description=description,
                        display=display,
                        server_params=list(server_populated or []),
                        depends_on=list(depends_on or []),
                        input_schema=input_schema,
                    )
                    return wrapped

                if func is not None and callable(func):
                    return _register(func)

                def decorator(fn: Callable):
                    return _register(fn)

                return decorator

            return _decorator

        # public decorators
        self.tool = _make_reg(display=False)  # type: ignore[attr-defined]
        self.display_tool = _make_reg(display=True)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Helper wrappers (copied/minimised from FastXMCPRouter)
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None = None):
        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls
        self._tool_formatters[tool_name] = formatter_instance

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
        _ft.update_wrapper(wrapper=wrapper, wrapped=func)  # type: ignore[arg-type]
        return wrapper

    # ------------------------------------------------------------------
    def _validate_payload(self, value: Any):  # noqa: D401, ANN001
        if isinstance(value, DisplayEnvelope):
            return value
        if isinstance(value, (dict, list)):
            try:
                DisplayEnvelope.model_validate(value)
            except Exception as exc:  # pragma: no cover
                raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail=str(exc))
            return value
        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            import xml.etree.ElementTree as _ET

            try:
                _ET.fromstring(value.lstrip())
                return value
            except Exception as exc:  # pragma: no cover
                raise XMCPError(code="INVALID_XML", retryable=False, detail=str(exc))
        raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail="Return value must be DisplayEnvelope or <tool_response> XML")

    # ------------------------------------------------------------------
    def _wrap_json(self, tool_name: str, xml_str: str, meta_extra: dict[str, Any] | None = None) -> str:
        import json as _json, time as _time
        meta = {"tool_name": tool_name, "ts": int(_time.time())}
        if meta_extra:
            meta.update(meta_extra)
        return _json.dumps({AI_MESSAGE_KEY: xml_str, "metadata": meta}, separators=(",", ":"))

    # ------------------------------------------------------------------
    def _build_error_xml(self, tool_name: str, exc: XMCPError) -> str:  # noqa: D401
        cddata = (exc.detail or exc.code).strip()
        meta_block = ""
        if exc.meta:
            import json as _json

            try:
                meta_block = f"\n    <meta>{_json.dumps(exc.meta, separators=(',', ':'))}</meta>"
            except Exception:
                pass
        xml = (
            f"<tool_response tool_name=\"{tool_name}\">\n"
            f"  <error code=\"{exc.code}\" retryable=\"{'true' if exc.retryable else 'false'}\">\n"
            f"    <![CDATA[{cddata}]]>{meta_block}\n  </error>\n"
            "</tool_response>"
        )

        if self._json_wrapper:
            return self._wrap_json(tool_name, xml, {"error": exc.code})
        return xml

    # ------------------------------------------------------------------
    def _serialize_success(self, tool_name: str, value: Any, formatter: Any | None):  # noqa: ANN001
        import json as _json

        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            return value

        display_block = ""
        llm_output = ""
        if isinstance(value, DisplayEnvelope):
            display_json = _json.dumps(value.dict_for_transport(), separators=(',', ':'))
            display_block = f"\n  <display><![CDATA[{display_json}]]></display>"
        else:
            if formatter is None:
                formatter = derive_formatter_from_data(sample=value) or DeclarativeXMLFormatter()
                self._tool_formatters[tool_name] = formatter
            formatted = formatter(value, indent_level=2)
            llm_output = f"\n  <llm_output><![CDATA[\n{formatted}\n  ]]></llm_output>"
        xml_out = f"<tool_response tool_name=\"{tool_name}\">{llm_output}{display_block}\n</tool_response>"

        if self._json_wrapper:
            return self._wrap_json(tool_name, xml_out)

        return xml_out

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, server_args: dict[str, Any] | None = None):
        """Execute *name* locally and return list[SimpleNamespace(text=str)]."""
        if name not in self._tools:
            raise XMCPError(code="UNKNOWN_TOOL", detail=name, retryable=False)

        info = self._tools[name]
        arguments = dict(arguments or {})
        server_args = server_args or {}

        # --------------------------------------------------------------
        # Authentication (Phase-1) – run backend once before execution
        # --------------------------------------------------------------

        # FastAPI passes a `Request` object – for in-process execution we
        # fabricate a minimal stub carrying a `state` attribute so backends
        # can attach principal information.  A *SimpleNamespace* is good
        # enough for this purpose.

        fake_request = SimpleNamespace(state=SimpleNamespace())
        principal = self._auth_backend.authenticate(request=fake_request)
        if inspect.isawaitable(object=principal):
            principal = await principal  # type: ignore[assignment]
        fake_request.state.principal = principal

        # Validate server_args vs allowed list -------------------------
        if server_args:
            for k in server_args:
                if k not in info.server_params:
                    raise XMCPError(code="UNKNOWN_SERVER_PARAM", detail=k, retryable=False)
                if k in arguments and self._strict_server_params:
                    raise XMCPError(code="FORBIDDEN_PARAM_OVERRIDE", detail=k, retryable=False)
            arguments = {**arguments, **server_args}

        # No provider injection yet (placeholder) ----------------------
        missing = [p for p in info.server_params if p not in arguments]
        if missing:
            raise XMCPError(code="MISSING_SERVER_PARAM", detail=missing[0], retryable=False)

        result_xml = await _maybe_await(info.func(**arguments))
        return [SimpleNamespace(text=result_xml)]

    # ------------------------------------------------------------------
    def list_tools(self):  # noqa: D401 – mimic MCP list_tools
        """Return a list of SimpleNamespace objects describing tools."""
        items: list[Any] = []
        for name, info in self._tools.items():
            ann: dict[str, Any] = {
                "is_display_tool": info.display,
            }
            if info.server_params:
                ann["server_populated"] = info.server_params
            if info.depends_on:
                ann["depends_on"] = info.depends_on
            items.append(
                SimpleNamespace(
                    name=name,
                    description=info.description,
                    inputSchema=info.input_schema,
                    annotations=ann,
                )
            )
        return items

    # ------------------------------------------------------------------
    def register_param_provider(self, fn: Callable):  # noqa: D401
        self._param_providers.append(fn)
        return fn


async def _maybe_await(v):
    if inspect.isawaitable(v):
        return await v
    return v 