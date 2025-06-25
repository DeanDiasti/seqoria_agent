from __future__ import annotations

"""Router support for FastXMCP.

This module introduces :class:`FastXMCPRouter`, an equivalent of FastAPI's
``APIRouter`` but aware of XMCP tool registration mechanics.  A router keeps
its own tool registry and can be mounted on a :class:`FastXMCP` application
via :pymeth:`fastapi.FastAPI.include_router`.
"""

import functools as _ft
import inspect
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Request

# Re-use internal helper types / errors / formatters from the main FastXMCP
from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.fastxmcp.app import _ToolInfo  # type: ignore
from seqoria_agent.server.formatters import (
    DeclarativeXMLFormatter,
)  # same default formatter as FastXMCP
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY  # central constant reuse
from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend  # local import to avoid hard dep

__all__: list[str] = ["FastXMCPRouter"]


class FastXMCPRouter(APIRouter):
    """Router variant that supports ``@tool`` / ``@display_tool`` decorators."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        prefix: str | None = "",
        tags: List[str] | None = None,
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        auth_backend=None,
        **kw,
    ):
        """Create a new router.

        Besides the regular ``APIRouter`` parameters, two XMCP-specific flags
        are accepted:

        Parameters
        ----------
        strict_mode:
            When *True*, every tool return value is validated and invalid
            payloads produce a structured ``<error>`` envelope.
        strict_server_params:
            If *True* (default) any attempt by the LLM/client to override a
            *server-populated* parameter is rejected with an error.
        json_wrapper:
            If *True*, the router will wrap JSON responses in a JSON envelope.
        """
        if auth_backend is None:
            auth_backend = NoAuthBackend()
        if not isinstance(auth_backend, BaseAuthBackend):
            raise TypeError("auth_backend must be an instance of BaseAuthBackend")

        self._auth_backend = auth_backend

        super().__init__(prefix=prefix or "", tags=tags or [], **kw)

        # ------------- XMCP-specific state ----------------------------
        self._strict_mode = strict_mode
        self._strict_server_params = bool(strict_server_params)
        self._json_wrapper = bool(json_wrapper)

        # Instance-local registries (merged into the FastXMCP app on mount)
        self._tools: Dict[str, _ToolInfo] = {}
        self._tool_formatters: dict[str, Any] = {}
        self._param_providers: list[Callable] = []

        # --------------------------------------------------------------
        # Helper: Python → JSON-schema type mapping
        # --------------------------------------------------------------

        def _json_schema_for_type(tp):  # noqa: ANN001
            from typing import get_origin

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

        # --------------------------------------------------------------
        # Decorator factory -------------------------------------------
        # --------------------------------------------------------------

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
                """Decorator for ``@router.tool``/``@router.display_tool``."""

                def _register(f: Callable):
                    # Wrap with error handling + validation first
                    formatter_cls = response_format if response_format is not None else xml_formatter
                    wrapped = self._wrap_tool_with_error_handling(
                        func=f, tool_name=f.__name__, formatter_cls=formatter_cls
                    )

                    # ------------------ Build JSON schema -------------
                    sig = inspect.signature(f)
                    props: dict[str, Any] = {}
                    required: list[str] = []

                    for pname, param in sig.parameters.items():
                        if pname in (server_populated or []):
                            continue  # hidden from LLM

                        annotation = (
                            param.annotation
                            if param.annotation is not inspect._empty  # type: ignore[attr-defined]
                            else str
                        )
                        props[pname] = _json_schema_for_type(annotation)

                        if param.default is inspect._empty:  # type: ignore[attr-defined]
                            required.append(pname)

                    input_schema = {"type": "object", "properties": props}
                    if required:
                        input_schema["required"] = required

                    # ------------------ Store registry entry ----------
                    self._tools[f.__name__] = _ToolInfo(
                        func=wrapped,
                        description=description,
                        display=display,
                        server_params=list(server_populated or []),
                        depends_on=list(depends_on or []),
                        input_schema=input_schema,
                    )

                    return wrapped

                # Support both ``@tool`` and ``@tool()`` syntaxes
                if func is not None and callable(func):
                    return _register(func)

                def decorator(fn: Callable):
                    return _register(fn)

                return decorator

            return _decorator

        # Public decorator factories
        self.tool = _make_reg(display=False)  # type: ignore[attr-defined]
        self.display_tool = _make_reg(display=True)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Shared helpers (mostly copied from FastXMCP)
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None = None):
        """Return a wrapper that adds XML wrapping & structured error handling."""
        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls

        self._tool_formatters[tool_name] = formatter_instance

        async def _async_impl(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
            try:
                result = await func(*args, **kwargs)  # type: ignore[arg-type]
                if self._strict_mode:
                    self._validate_payload(value=result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        def _sync_impl(*args: Any, **kwargs: Any):  # noqa: ANN401
            try:
                result = func(*args, **kwargs)
                if self._strict_mode:
                    self._validate_payload(value=result)
                return self._serialize_success(tool_name, result, formatter_instance)
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        wrapper = _async_impl if inspect.iscoroutinefunction(func) else _sync_impl
        _ft.update_wrapper(wrapper=wrapper, wrapped=func)  # type: ignore[arg-type]
        return wrapper

    # ------------------------------------------------------------------
    def _validate_payload(self, value: Any):  # noqa: D401, ANN001
        """Strict-mode payload validation (copy of FastXMCP logic)."""
        if isinstance(value, DisplayEnvelope):
            return value

        if isinstance(value, (dict, list)):
            # Try coercing into DisplayEnvelope
            try:
                DisplayEnvelope.model_validate(value)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail=str(exc))
            return value

        if isinstance(value, str):
            xml_s = value.lstrip()
            if xml_s.startswith("<tool_response"):
                import xml.etree.ElementTree as _ET

                try:
                    _ET.fromstring(xml_s)
                    return value
                except Exception as exc:  # pragma: no cover
                    raise XMCPError(code="INVALID_XML", retryable=False, detail=str(exc))

        raise XMCPError(
            code="INVALID_PAYLOAD",
            retryable=False,
            detail="Return value must be DisplayEnvelope or <tool_response> XML",
        )

    # ------------------------------------------------------------------
    def _build_error_xml(self, tool_name: str, exc: XMCPError) -> str:  # noqa: D401
        cddata = (exc.detail or exc.code).strip()
        meta_block = ""
        if exc.meta:
            import json as _json

            try:
                meta_json = _json.dumps(exc.meta, separators=(",", ":"))
                meta_block = f"\n    <meta>{meta_json}</meta>"
            except Exception:  # pragma: no cover
                pass

        xml = (
            f"<tool_response tool_name=\"{tool_name}\">\n"
            f"  <error code=\"{exc.code}\" retryable=\"{'true' if exc.retryable else 'false'}\">\n"
            f"    <![CDATA[{cddata}]]>" + meta_block + "\n  </error>\n"
            "</tool_response>"
        )

        if self._json_wrapper:
            return self._wrap_json(tool_name, xml, {"error": exc.code})
        return xml

    # ------------------------------------------------------------------
    def _wrap_json(self, tool_name: str, xml_str: str, meta_extra: dict[str, Any] | None = None) -> str:
        import json as _json, time as _time
        meta = {"tool_name": tool_name, "ts": int(_time.time())}
        if meta_extra:
            meta.update(meta_extra)
        return _json.dumps({AI_MESSAGE_KEY: xml_str, "metadata": meta}, separators=(",", ":"))

    # ------------------------------------------------------------------
    def _serialize_success(self, tool_name: str, value: Any, formatter: Any | None):  # noqa: ANN001
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
                from seqoria_agent.server.formatters import derive_formatter_from_data
                formatter = derive_formatter_from_data(sample=value) or DeclarativeXMLFormatter()
                self._tool_formatters[tool_name] = formatter
            formatted = formatter(value, indent_level=2)
            llm_output_xml = f"\n  <llm_output><![CDATA[\n{formatted}\n  ]]></llm_output>"

        xml_out = f"<tool_response tool_name=\"{tool_name}\">{llm_output_xml}{display_block}\n</tool_response>"

        if self._json_wrapper:
            return self._wrap_json(tool_name=tool_name, xml_str=xml_out)

        return xml_out

    # ------------------------------------------------------------------
    async def _inject_server_params(
        self,
        *,
        request: Request,
        tool_name: str,
        info: _ToolInfo,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:  # noqa: D401
        """Return a new *arguments* dict with server parameters injected."""
        injected: dict[str, Any] = dict(arguments)  # shallow copy

        for param in info.server_params:
            if param in injected:
                continue  # already supplied (serverArgs merge)

            value_found = False
            for prov in self._param_providers:
                try:
                    res = prov(request, tool_name, param)
                    if inspect.isawaitable(res):  # type: ignore[arg-type]
                        res = await res  # type: ignore[assignment]
                except Exception:  # pragma: no cover
                    continue

                if res is not None:
                    injected[param] = res
                    value_found = True
                    break

            if not value_found:
                raise XMCPError(code="MISSING_SERVER_PARAM", detail=param, retryable=False)

        return injected

    # ------------------------------------------------------------------
    # Public API: parameter provider registration ----------------------
    # ------------------------------------------------------------------

    def register_param_provider(self, fn: Callable):  # noqa: D401
        """Register *fn* as a server parameter provider for this router."""
        self._param_providers.append(fn)
        return fn 