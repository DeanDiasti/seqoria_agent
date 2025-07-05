from __future__ import annotations
# pragma: no cover

"""Framework-agnostic core mechanics shared by FastXMCP, FastXMCPRouter and
LocalToolbox.

The goal of *ToolCore* is to centralise **tool registration** and **execution**
so the public facades can delegate without copying hundreds of lines.

Phase-1 only – parameter injection & auth are handled by the caller just like
before.  ToolCore focuses on:
• @tool / @display_tool decorator factories with JSON-schema generation
• duplicate-name detection
• runtime wrappers that serialise return values into the <tool_response> XML
• strict-mode payload validation
• optional JSON wrapper (ai_message envelope)
"""

from typing import Any, Callable, Dict, Sequence, get_origin
import inspect
import functools as _ft
import json as _json
import logging
import time as _time

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.formatters import BaseXMLFormatter  # only for type hints
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY
from seqoria_agent.server.xmcp_core.formatter import get_formatter

from .models import ToolInfo
from .param_injection import ParamInjector

__all__: list[str] = ["ToolCore"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_await(val):  # type: ignore[return-value, arg-type]
    if inspect.isawaitable(object=val):
        return val  # caller awaits
    async def _wrapper() -> Any:
        return val
    return _wrapper()


def _json_schema_for_type(tp):  # noqa: ANN001
    """Return minimal JSON-schema for *tp* covering basic scalars.

    We cannot rely on simple ``{int: 'integer', bool: 'boolean'}`` mapping
    because ``bool`` is a *sub-class* of ``int`` **and** considered equal as
    a dict key, so the later insertion would override the first.  Use an
    explicit ``if/elif`` ladder instead.
    """
    origin = get_origin(tp) or tp

    if origin is bool:
        return {"type": "boolean"}
    if origin in (int,):
        return {"type": "integer"}
    if origin in (float,):
        return {"type": "number"}
    if origin is str:
        return {"type": "string"}
    # Annotations may be *stringified* when PEP-649 postponed evaluation is
    # enabled (``from __future__ import annotations``).  Handle the common
    # built-in scalars here so unit tests relying on future annotations pass.
    if isinstance(origin, str):
        lowered = origin.lower()
        if lowered in {"int", "integer"}:
            return {"type": "integer"}
        if lowered in {"bool", "boolean"}:
            return {"type": "boolean"}
        if lowered in {"float", "number"}:
            return {"type": "number"}
        return {"type": "string"}
    # fallback – treat as string
    return {"type": "string"}


# ---------------------------------------------------------------------------
# Optional Pydantic integration – enrich parameter schema via Field metadata
# ---------------------------------------------------------------------------
try:
    # Pydantic v1 & v2 expose FieldInfo in the same sub-module path
    from pydantic.fields import FieldInfo as _PydanticFieldInfo  # type: ignore
    from pydantic import create_model as _pyd_create_model  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – pydantic not installed
    _PydanticFieldInfo = None  # type: ignore
    _pyd_create_model = None  # type: ignore


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

class ToolCore:
    """Registry & execution engine for XMCP tools (framework agnostic)."""

    # public decorator attributes – initialised in __init__
    tool: Callable  # filled in at runtime
    display_tool: Callable  # dito

    def __init__(
        self,
        *,
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._strict_mode = bool(strict_mode)
        self._strict_server_params = bool(strict_server_params)
        self._json_wrapper = bool(json_wrapper)
        self._log = logger or logging.getLogger(name=__name__)

        # registries -----------------------------------------------------
        self._tools: Dict[str, ToolInfo] = {}
        self._tool_formatters: dict[str, Any] = {}
        self._param_providers: list[Callable] = []  # Phase-2

        # create decorator factories ------------------------------------
        self.tool = self._make_reg(display=False)
        self.display_tool = self._make_reg(display=True)

        # ------------------------------------------------------------------
        # Parameter injection service (Phase-2) ----------------------------
        # ------------------------------------------------------------------
        self._injector = ParamInjector(providers=self._param_providers, strict_mode=self._strict_server_params)

    # ------------------------------------------------------------------
    # Public accessors used by caller wrappers
    # ------------------------------------------------------------------

    @property
    def tools(self) -> Dict[str, ToolInfo]:  # noqa: D401 – quick accessor
        return self._tools

    # ------------------------------------------------------------------
    # Decorator factory
    # ------------------------------------------------------------------

    def _make_reg(self, *, display: bool):  # noqa: D401
        """Return @tool / @display_tool decorator."""

        def _decorator(
            func: Callable | None = None,
            *,
            xml_formatter: Any | None = None,
            response_format: Any | None = None,
            description: str | None = None,
            server_populated: Sequence[str] | None = None,
            depends_on: Sequence[str] | None = None,
        ):
            def _register(f: Callable):
                name = f.__name__
                if name in self._tools:
                    raise XMCPError(code="DUPLICATE_TOOL", detail=name, retryable=False)

                # pick a formatter class if explicitly passed
                fmt_cls = response_format if response_format is not None else xml_formatter
                wrapped = self._wrap_tool_with_error_handling(f, name, fmt_cls)

                # generate simple parameter JSON-schema
                sig = inspect.signature(f)
                props: dict[str, Any] = {}
                required: list[str] = []
                for p_name, param in sig.parameters.items():
                    # ------------------------------------------------------------------
                    # Skip server-populated parameters – they are hidden from the LLM
                    # ------------------------------------------------------------------
                    if p_name in (server_populated or []):
                        continue

                    ann = param.annotation if param.annotation is not inspect._empty else str

                    # ------------------------------------------------------------------
                    # Pydantic Field – leverage its rich JSON-schema generation
                    # ------------------------------------------------------------------
                    if (
                        _PydanticFieldInfo is not None and  # pydantic import succeeded
                        _pyd_create_model is not None and
                        param.default is not inspect._empty and
                        isinstance(param.default, _PydanticFieldInfo)
                    ):
                        # Build a tiny throw-away model so pydantic can derive the
                        # correct JSON Schema for us (including constraints & docs).
                        TmpModel = _pyd_create_model(  # type: ignore
                            "_Tmp", **{p_name: (ann, param.default)}
                        )

                        if hasattr(TmpModel, "model_json_schema"):
                            # Pydantic ≥2
                            fld_schema = TmpModel.model_json_schema()["properties"][p_name]  # type: ignore[index]
                        else:
                            # Pydantic v1 – .schema()
                            fld_schema = TmpModel.schema()["properties"][p_name]  # type: ignore[attr-defined]

                        props[p_name] = fld_schema

                        # FieldInfo helper – works across Pydantic versions.
                        try:
                            if param.default.is_required():  # type: ignore[attr-defined]
                                required.append(p_name)
                        except Exception:
                            # Fallback check based on sentinel values
                            if getattr(param.default, "default", Ellipsis) is Ellipsis:
                                required.append(p_name)

                        continue  # handled – next parameter

                    # ------------------------------------------------------------------
                    # Fallback to primitive scalar mapping (existing behaviour)
                    # ------------------------------------------------------------------
                    props[p_name] = _json_schema_for_type(ann)
                    if param.default is inspect._empty:
                        required.append(p_name)
                input_schema: dict[str, Any] = {"type": "object", "properties": props}
                if required:
                    input_schema["required"] = required

                self._tools[name] = ToolInfo(
                    func=wrapped,
                    description=description,
                    display=display,
                    server_params=list(server_populated or []),
                    depends_on=list(depends_on or []),
                    input_schema=input_schema,
                )
                return wrapped

            # support both @tool and @tool(...)
            if func is not None and callable(func):
                return _register(func)
            return _register

        return _decorator

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None):
        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls
        self._tool_formatters[tool_name] = formatter_instance

        async def _async(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = await func(*args, **kwargs)  # type: ignore[arg-type]
                if self._strict_mode:
                    self._validate_payload(result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        def _sync(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = func(*args, **kwargs)
                if self._strict_mode:
                    self._validate_payload(result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        wrapper = _async if inspect.iscoroutinefunction(func) else _sync
        _ft.update_wrapper(wrapper=wrapper, wrapped=func)  # type: ignore[arg-type]
        return wrapper

    # ------------------------------------------------------------------
    def _validate_payload(self, value: Any):  # noqa: D401, ANN001
        if isinstance(value, DisplayEnvelope):
            return value
        if isinstance(value, (dict, list)):
            try:
                DisplayEnvelope.model_validate(value)  # type: ignore[arg-type]
            except Exception as exc:
                raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail=str(exc))
            return value
        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            import xml.etree.ElementTree as _ET
            try:
                _ET.fromstring(value.lstrip())
                return value
            except Exception as exc:
                raise XMCPError(code="INVALID_XML", retryable=False, detail=str(exc))
        raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail="Return value must be DisplayEnvelope or <tool_response> XML")

    # ------------------------------------------------------------------
    def _wrap_json(self, tool_name: str, xml_str: str, meta_extra: dict[str, Any] | None = None):  # noqa: ANN001
        meta = {"tool_name": tool_name, "ts": int(_time.time())}
        if meta_extra:
            meta.update(meta_extra)
        return _json.dumps({AI_MESSAGE_KEY: xml_str, "metadata": meta}, separators=(",", ":"))

    # ------------------------------------------------------------------
    def _build_error_xml(self, tool_name: str, exc: XMCPError):  # noqa: D401, ANN001
        cddata = (exc.detail or exc.code).strip()
        meta_block = ""
        if exc.meta:
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
        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            return value

        display_block = ""
        llm_output_block = ""
        if isinstance(value, DisplayEnvelope):
            display_block = f"\n  <display><![CDATA[{_json.dumps(value.dict_for_transport(), separators=(',', ':'))}]]></display>"
        else:
            if formatter is None:
                formatter = get_formatter(value, strict_mode=self._strict_mode)
            formatted = formatter(value, indent_level=2)
            llm_output_block = f"\n  <llm_output><![CDATA[\n{formatted}\n  ]]></llm_output>"
        xml_out = f"<tool_response tool_name=\"{tool_name}\">{llm_output_block}{display_block}\n</tool_response>"
        if self._json_wrapper:
            return self._wrap_json(tool_name, xml_out)
        return xml_out

    # ------------------------------------------------------------------
    # Public runtime API ------------------------------------------------
    # ------------------------------------------------------------------

    async def run_tool(self, name: str, arguments: Dict[str, Any] | None = None, *, server_args: Dict[str, Any] | None = None):  # noqa: D401
        """Execute *name* and return the XML response string.

        Phase-1 does **not** implement advanced server parameter injection –
        that is still handled by the caller (FastXMCP / Router / LocalToolbox).
        """
        if name not in self._tools:
            raise XMCPError(code="UNKNOWN_TOOL", detail=name, retryable=False)

        info = self._tools[name]
        arguments = dict(arguments or {})
        server_args = server_args or {}

        if server_args:
            for k in server_args:
                if k not in info.server_params:
                    raise XMCPError(code="UNKNOWN_SERVER_PARAM", detail=k, retryable=False)
                if k in arguments and self._strict_server_params:
                    raise XMCPError(code="FORBIDDEN_PARAM_OVERRIDE", detail=k, retryable=False)
            arguments = {**arguments, **server_args}

        return await _maybe_await(val=info.func(**arguments))

    # ------------------------------------------------------------------
    # Placeholder – ParamInjector arrives in Phase-2 --------------------
    # ------------------------------------------------------------------
    def register_param_provider(self, fn: Callable):  # noqa: D401
        self._param_providers.append(fn)
        return fn

    # Public helper – delegate to ParamInjector ------------------------
    async def inject_server_params(self,
                                   *,
                                   request_or_ns: Any,
                                   tool_name: str,
                                   info: ToolInfo,
                                   arguments: dict[str, Any] | None = None,
                                   server_args: dict[str, Any] | None = None) -> Dict[str, Any]:  # noqa: D401
        """Proxy to :pyclass:`ParamInjector.inject`."""
        return await self._injector.inject(request_or_ns=request_or_ns,
                                           tool_name=tool_name,
                                           info=info,
                                           arguments=arguments,
                                           server_args=server_args) 