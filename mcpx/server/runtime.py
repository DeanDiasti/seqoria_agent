"""MCPx server runtime - enhanced wrapper around :class:`fastmcp.FastMCP`.

This class replaces the legacy ``xmcp.fastxmcp.XMCP`` while keeping the public
behaviour intact.  It adds specialised decorators (``@display_tool``) and
structured error handling that converts :class:`mcpx.models.errors.XMCPError`
into the canonical `<error>` XML envelope understood by
``mcpx.client.AIClient``.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional

from fastmcp import FastMCP
from mcpx.models.envelope import DisplayEnvelope
import json as _json

from mcpx.models.errors import XMCPError

__all__: list[str] = [
    "XMCP",
]


class XMCP(FastMCP):
    """XMCP server with first-class support for specialised tool flavours.

    This class extends `fastmcp.FastMCP` to provide a richer environment for
    building tool-augmented AI applications. It introduces the concept of
    "display tools" for sending rich UI components to a user, structured
    error handling, and a pluggable XML formatting system.

    Attributes
    ----------
    registered_tools : List[Dict[str, Any]]
        A list of tool functions registered with the server.
    registered_resources : List[Dict[str, Any]]
        A list of resource functions registered with the server.
    registered_prompts : List[Dict[str, Any]]
        A list of prompt functions registered with the server.
    """

    def __init__(self, *args: Any, strict_mode: bool = False, **kwargs: Any) -> None:
        """Create an XMCP runtime.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the `fastmcp.FastMCP` parent class.
        strict_mode : bool, optional
            If `True`, the return value of every tool is validated against
            either the `DisplayEnvelope` schema or a well-formed XML structure.
            Invalid payloads are converted into a structured `XMCPError`.
            Defaults to `False`.
        **kwargs : Any
            Keyword arguments passed to the `fastmcp.FastMCP` parent class.
        """

        self._strict_mode = strict_mode
        super().__init__(*args, **kwargs)
        self.registered_tools: List[Dict[str, Any]] = []
        self.registered_resources: List[Dict[str, Any]] = []
        self.registered_prompts: List[Dict[str, Any]] = []

        # Map tool_name -> XML formatter instance (or None for default)
        self._tool_formatters: dict[str, Any] = {}

    def tool(self, func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        """A decorator for registering a standard tool.

        Standard tools return data that is intended to be processed by the LLM
        in a subsequent step. The return value is automatically wrapped in a
        `<tool_response>` XML structure.

        This decorator can be used with or without arguments.

        Parameters
        ----------
        func : Optional[Callable], optional
            The function to decorate. This is supplied automatically when the
            decorator is used without parentheses. Defaults to `None`.
        xml_formatter : Optional[BaseXMLFormatter], optional
            A formatter instance to control how the tool's output is serialized
            into XML for the LLM. If not provided, a default formatter is used.
            This is passed as a keyword argument.

        Returns
        -------
        Callable
            The decorated tool function.
        """
        xml_formatter_cls = kwargs.pop("xml_formatter", None)

        if func:
            # Used as @mcp.tool
            self.registered_tools.append({"func": func, "name": func.__name__, "kwargs": kwargs})

            wrapped = self._wrap_tool_with_error_handling(func, func.__name__, xml_formatter_cls)
            return super().tool(name_or_fn=wrapped)
        else:
            # Used as @mcp.tool()
            original_decorator = super().tool(**kwargs)

            def decorator(f: Callable) -> Callable:
                self.registered_tools.append({"func": f, "name": f.__name__, "kwargs": kwargs})
                wrapped = self._wrap_tool_with_error_handling(f, f.__name__, xml_formatter_cls)
                return original_decorator(wrapped)

            return decorator

    def resources(self, func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        """A decorator for defining resources available to tools.

        This is a pass-through to the underlying `fastmcp.FastMCP.resources`
        decorator and behaves identically. It is used to define functions
        that provide resources (like database connections or API clients)
        that can be injected into tools.

        Parameters
        ----------
        func : Optional[Callable], optional
            The function to decorate. Defaults to `None`.
        **kwargs : Any
            Keyword arguments passed to the parent decorator.

        Returns
        -------
        Callable
            The decorated resource function.
        """
        if func:
            # Used as @mcp.resources
            self.registered_resources.append({"func": func, "name": func.__name__, "kwargs": kwargs})
            return super().resources(func)
        else:
            # Used as @mcp.resources()
            original_decorator = super().resources(**kwargs)

            def decorator(f: Callable) -> Callable:
                self.registered_resources.append({"func": f, "name": f.__name__, "kwargs": kwargs})
                return original_decorator(f)

            return decorator

    def prompt(self, func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        """A decorator for defining a system prompt.

        This is a pass-through to the underlying `fastmcp.FastMCP.prompt`
        decorator. It registers a function that generates the system prompt
        for the LLM.

        Parameters
        ----------
        func : Optional[Callable], optional
            The function to decorate. Defaults to `None`.
        **kwargs : Any
            Keyword arguments passed to the parent decorator.

        Returns
        -------
        Callable
            The decorated prompt function.
        """
        if func:
            # Used as @mcp.prompt
            self.registered_prompts.append({"func": func, "name": func.__name__, "kwargs": kwargs})
            return super().prompt(name_or_fn=func)
        else:
            # Used as @mcp.prompt()
            original_decorator = super().prompt(**kwargs)

            def decorator(f: Callable) -> Callable:
                self.registered_prompts.append({"func": f, "name": f.__name__, "kwargs": kwargs})
                return original_decorator(f)

            return decorator

    def display_tool(self, func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        """Decorator for creating a tool that can send rich UI to a user.

        Display tools are "hybrid" tools. They can either return data for the
        LLM to process (like a standard tool) or return a `DisplayEnvelope`
        object to be rendered directly by a user interface. This behaviour is
        typically controlled by a boolean `show_user` argument in the tool's
        signature.

        The `AIClient` recognizes display tools and can short-circuit the
        LLM reasoning loop when a `DisplayEnvelope` is returned.

        Parameters
        ----------
        func : Optional[Callable], optional
            The function to decorate. Defaults to `None`.
        **kwargs : Any
            Keyword arguments passed to the underlying `fastmcp.tool` decorator.

        Returns
        -------
        Callable
            The decorated display tool function.
        """

        # Display tools ignore xml_formatter on purpose
        if func is not None:
            # Used as ``@mcp.display_tool``
            self.registered_tools.append({"func": func, "name": func.__name__, "kwargs": kwargs})
            wrapped = self._wrap_tool_with_error_handling(func, func.__name__, None)
            return super().tool(name_or_fn=wrapped, annotations={"is_display_tool": True})

        # Used as ``@mcp.display_tool(...)`` allowing custom FastMCP kwargs.
        original_decorator = super().tool(annotations={"is_display_tool": True}, **kwargs)

        def decorator(f: Callable) -> Callable:
            self.registered_tools.append({"func": f, "name": f.__name__, "kwargs": kwargs})
            wrapped = self._wrap_tool_with_error_handling(f, f.__name__, None)
            return original_decorator(wrapped)

        return decorator

    def _wrap_tool_with_error_handling(self, func: Callable, tool_name: str, formatter_cls: Any | None = None) -> Callable:
        """Return a wrapper that converts :class:`XMCPError` into XML payload.

        This keeps the *public* tool API unchanged while giving us a single
        interception point for structured error handling.
        """

        from mcpx.server.formatters import DeclarativeXMLFormatter  # local import

        formatter_instance = None
        if formatter_cls is not None:
            formatter_instance = formatter_cls() if isinstance(formatter_cls, type) else formatter_cls
        self._tool_formatters[tool_name] = formatter_instance

        if inspect.iscoroutinefunction(func):

            async def _async_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                try:
                    result = await func(*args, **kwargs)
                    if self._strict_mode:
                        self._validate_payload(value=result)

                    # Convert success payload into XML for non-display tools
                    xml_str = self._serialize_success(tool_name, result, formatter_instance)
                    return xml_str
                except XMCPError as exc:
                    return self._build_error_xml(tool_name, exc)

            functools.update_wrapper(_async_wrapper, func)
            _async_wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
            return _async_wrapper

        # Synchronous fallback -------------------------------------------------
        def _sync_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
            try:
                result = func(*args, **kwargs)
                if self._strict_mode:
                    self._validate_payload(value=result)

                xml_str = self._serialize_success(tool_name, result, formatter_instance)
                return xml_str
            except XMCPError as exc:
                return self._build_error_xml(tool_name, exc)

        functools.update_wrapper(_sync_wrapper, func)
        _sync_wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        return _sync_wrapper

    # ------------------------------------------------------------------
    # Strict-mode payload validator (opt-in)
    # ------------------------------------------------------------------

    def _validate_payload(self, value: Any) -> Any:
        """Ensure *value* is DisplayEnvelope/compatible or valid XML."""

        from mcpx.models.envelope import DisplayEnvelope  # local import to avoid cycle

        # 1️⃣ DisplayEnvelope instance or dict-like
        if isinstance(value, DisplayEnvelope):
            return value  # already validated by pydantic

        if isinstance(value, (dict, list)):
            # Try to coerce into DisplayEnvelope; on failure raise XMCPError
            try:
                DisplayEnvelope.model_validate(obj=value)
            except Exception as exc:
                raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail=str(object=exc))
            return value

        # 2️⃣ XML string produced by regular @tool
        if isinstance(value, str):
            xml_s = value.lstrip()
            if xml_s.startswith("<tool_response"):
                # rudimentary well-formedness check
                import xml.etree.ElementTree as _ET
                try:
                    _ET.fromstring(xml_s)
                except Exception as exc:  # Malformed
                    raise XMCPError(code="INVALID_XML", retryable=False, detail=str(exc))
                return value

        # Everything else → invalid
        raise XMCPError(code="INVALID_PAYLOAD", retryable=False, detail="Return value must be DisplayEnvelope or <tool_response> XML")

    # --------------------------------------------------------------
    # Helper – error XML builder (shared by sync/async paths)
    # --------------------------------------------------------------
    @staticmethod
    def _build_error_xml(tool_name: str, exc: "XMCPError") -> str:  # noqa: D401
        """Serialize *exc* into the canonical `<error>` XML snippet."""

        cddata = (exc.detail or exc.code).strip()
        meta_block = ""
        if exc.meta:
            import json as _json  # local import to avoid top-level cost

            try:
                meta_json = _json.dumps(exc.meta, separators=(",", ":"))
                meta_block = f"\n    <meta>{meta_json}</meta>"
            except Exception:
                # Best-effort – ignore meta if not serialisable
                pass

        return (
            f"<tool_response tool_name=\"{tool_name}\">\n"
            f"  <error code=\"{exc.code}\" retryable=\"{'true' if exc.retryable else 'false'}\">\n"
            f"    <![CDATA[{cddata}]]>" + meta_block + "\n  </error>\n"
            f"</tool_response>"
        )

    # --------------------------------------------------------------
    # Success serialization helper
    # --------------------------------------------------------------

    def _serialize_success(self, tool_name: str, value: Any, formatter: Any | None) -> str:  # noqa: ANN401
        """Return `<tool_response>` XML for *value* using *formatter*."""

        if isinstance(value, str) and value.lstrip().startswith("<tool_response"):
            return value  # assume already formatted

        llm_output_xml: str
        display_block: str = ""

        if isinstance(value, DisplayEnvelope):
            # Keep an untouched display block; llm_output stays empty
            display_json = _json.dumps(value.dict_for_transport(), separators=(",", ":"))
            display_block = f"\n  <display><![CDATA[{display_json}]]></display>"
            llm_output_xml = ""
        else:
            # Format value via formatter or default fall-back
            if formatter is None:
                from mcpx.server.formatters import DeclarativeXMLFormatter  # local import
                formatter = DeclarativeXMLFormatter()

            formatted = formatter(value, indent_level=2)
            llm_output_xml = f"\n  <llm_output><![CDATA[\n{formatted}\n  ]]></llm_output>"

        return f"<tool_response tool_name=\"{tool_name}\">{llm_output_xml}{display_block}\n</tool_response>"
