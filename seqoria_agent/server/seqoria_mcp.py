"""SeqoriaMCP - Extended MCP FastMCP server with Seqoria features.

This module provides SeqoriaMCP, an extension of the official MCP SDK's FastMCP
that adds Seqoria-specific features:

- Server-populated parameters (hidden from LLM, injected at runtime)
- Tool dependency annotations (depends_on)
- Authentication backend integration
- Response formatting (XML envelope format)

Example
-------
>>> from seqoria_agent.server import SeqoriaMCP
>>>
>>> mcp = SeqoriaMCP("MyServer")
>>>
>>> @mcp.tool(server_populated=["user_id"])
>>> async def greet(name: str, user_id: str) -> str:
...     return f"Hello {name}! (user: {user_id})"
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from mcp.server.fastmcp import Context, FastMCP

from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend

__all__ = [
    "SeqoriaMCP",
    "ToolMetadata",
]

F = TypeVar("F", bound=Callable[..., Any])


class ToolMetadata:
    """Metadata for a registered tool.

    Attributes
    ----------
    name : str
        The tool function name.
    server_populated : list[str]
        Parameter names that are server-populated (hidden from LLM).
    depends_on : list[str]
        Names of tools that must execute before this one.
    description : str | None
        Tool description (from docstring or explicit).
    """

    __slots__ = ("name", "server_populated", "depends_on", "description")

    def __init__(
        self,
        name: str,
        *,
        server_populated: list[str] | None = None,
        depends_on: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.server_populated = list(server_populated or [])
        self.depends_on = list(depends_on or [])
        self.description = description


class SeqoriaMCP(FastMCP):
    """Extended MCP FastMCP with Seqoria-specific features.

    Parameters
    ----------
    name : str
        Server name for MCP identification.
    auth_backend : BaseAuthBackend | None
        Authentication backend for validating requests.
    strict_server_params : bool
        If True, raise error when LLM tries to provide server-populated params.
    json_response : bool
        Whether to use JSON response format (passed to FastMCP).
    **kwargs
        Additional arguments passed to FastMCP.

    Example
    -------
    >>> mcp = SeqoriaMCP("MyServer")
    >>>
    >>> @mcp.tool(server_populated=["tenant_id"], depends_on=["auth_check"])
    >>> async def get_data(query: str, tenant_id: str) -> dict:
    ...     return {"result": query, "tenant": tenant_id}
    """

    def __init__(
        self,
        name: str = "SeqoriaMCP",
        *,
        auth_backend: BaseAuthBackend | None = None,
        strict_server_params: bool = True,
        json_response: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, json_response=json_response, **kwargs)

        # Authentication
        self._auth_backend: BaseAuthBackend = auth_backend or NoAuthBackend()

        # Configuration
        self._strict_server_params = strict_server_params

        # Tool metadata registry
        self._tool_metadata: dict[str, ToolMetadata] = {}

        # Parameter providers for server-populated params
        self._param_providers: list[Callable[..., Any]] = []

    @property
    def tool_metadata(self) -> dict[str, ToolMetadata]:
        """Return the tool metadata registry (read-only view)."""
        return dict(self._tool_metadata)

    def register_param_provider(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a parameter provider for server-populated parameters.

        The provider is called with (ctx, tool_name, param_name) and should
        return the value or None if it cannot provide the parameter.

        Parameters
        ----------
        fn : Callable
            Provider function with signature: (ctx, tool_name, param_name) -> Any | None

        Returns
        -------
        Callable
            The same function (allows use as decorator).
        """
        self._param_providers.append(fn)
        return fn

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        server_populated: Sequence[str] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> Callable[[F], F]:
        """Register a tool with extended Seqoria features.

        This decorator extends the MCP FastMCP @tool decorator to add:
        - server_populated: Parameters hidden from LLM, injected at runtime
        - depends_on: Tool dependencies for execution ordering

        Parameters
        ----------
        name : str | None
            Tool name (defaults to function name).
        description : str | None
            Tool description (defaults to docstring).
        server_populated : Sequence[str] | None
            Parameter names that are server-populated (hidden from LLM schema).
        depends_on : Sequence[str] | None
            Names of tools that must execute before this one.

        Returns
        -------
        Callable
            Decorator function.

        Example
        -------
        >>> @mcp.tool(server_populated=["user_id"])
        >>> async def my_tool(name: str, user_id: str, ctx: Context) -> str:
        ...     return f"Hello {name} (user: {user_id})"
        """
        server_populated_list = list(server_populated or [])
        depends_on_list = list(depends_on or [])

        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]

            # Store metadata
            self._tool_metadata[tool_name] = ToolMetadata(
                name=tool_name,
                server_populated=server_populated_list,
                depends_on=depends_on_list,
                description=tool_desc,
            )

            # Create wrapper that injects server-populated params
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract context if present
                ctx = kwargs.get("ctx")
                if ctx is None:
                    for arg in args:
                        if isinstance(arg, Context):
                            ctx = arg
                            break

                # Inject server-populated parameters
                if server_populated_list and ctx is not None:
                    for param in server_populated_list:
                        if param not in kwargs:
                            value = await self._resolve_server_param(
                                ctx, tool_name, param
                            )
                            if value is not None:
                                kwargs[param] = value

                # Call original function
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result

                return result

            # Build modified schema that excludes server-populated params
            modified_fn = self._create_schema_modified_function(
                fn, wrapper, server_populated_list, depends_on_list
            )

            # Register with parent FastMCP using the base decorator
            # We need to call the parent's tool method properly
            parent_decorator = super(SeqoriaMCP, self).tool(
                name=tool_name,
                description=tool_desc,
            )
            parent_decorator(modified_fn)

            return fn  # type: ignore[return-value]

        return decorator

    def _create_schema_modified_function(
        self,
        original_fn: Callable[..., Any],
        wrapper_fn: Callable[..., Any],
        server_populated: list[str],
        depends_on: list[str],
    ) -> Callable[..., Any]:
        """Create a function with modified signature for schema generation.

        This creates a wrapper that has the server-populated parameters
        removed from its signature, so the MCP SDK generates a schema
        that hides these parameters from the LLM.
        """
        if not server_populated:
            return wrapper_fn

        # Get original signature
        sig = inspect.signature(original_fn)
        params = list(sig.parameters.values())

        # Filter out server-populated params
        filtered_params = [
            p for p in params if p.name not in server_populated and p.name != "ctx"
        ]

        # Add ctx as optional at the end if the original had it
        has_ctx = any(p.name == "ctx" for p in params)
        if has_ctx:
            ctx_param = inspect.Parameter(
                "ctx",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Context,
            )
            filtered_params.append(ctx_param)

        # Create new signature
        new_sig = sig.replace(parameters=filtered_params)

        # Create a new function with the modified signature
        @functools.wraps(original_fn)
        async def schema_fn(*args: Any, **kwargs: Any) -> Any:
            return await wrapper_fn(*args, **kwargs)

        schema_fn.__signature__ = new_sig  # type: ignore[attr-defined]

        # Add annotations metadata for client-side awareness
        if not hasattr(schema_fn, "__mcp_annotations__"):
            schema_fn.__mcp_annotations__ = {}  # type: ignore[attr-defined]
        schema_fn.__mcp_annotations__["server_populated"] = server_populated  # type: ignore[attr-defined]
        schema_fn.__mcp_annotations__["depends_on"] = depends_on  # type: ignore[attr-defined]

        return schema_fn

    async def _resolve_server_param(
        self,
        ctx: Context,
        tool_name: str,
        param_name: str,
    ) -> Any | None:
        """Resolve a server-populated parameter value.

        Resolution order:
        1. Built-in params (principal, user_id)
        2. Custom param providers
        3. Request context properties

        Parameters
        ----------
        ctx : Context
            The MCP request context.
        tool_name : str
            Name of the tool requesting the parameter.
        param_name : str
            Name of the parameter to resolve.

        Returns
        -------
        Any | None
            The resolved value or None.
        """
        # Try built-in parameters first
        if param_name == "principal":
            return self._get_principal_from_context(ctx)

        if param_name == "user_id":
            principal = self._get_principal_from_context(ctx)
            if principal is not None:
                if hasattr(principal, "id"):
                    return getattr(principal, "id")
                elif isinstance(principal, dict) and "id" in principal:
                    return principal["id"]

        # Try custom providers
        for provider in self._param_providers:
            try:
                result = provider(ctx, tool_name, param_name)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    return result
            except Exception:
                continue

        # Try request context
        if hasattr(ctx, "request_context"):
            req_ctx = ctx.request_context
            if hasattr(req_ctx, param_name):
                return getattr(req_ctx, param_name)
            if isinstance(req_ctx, dict) and param_name in req_ctx:
                return req_ctx[param_name]

        return None

    def _get_principal_from_context(self, ctx: Context) -> Any | None:
        """Extract the authenticated principal from MCP context."""
        # Check various locations where principal might be stored
        if hasattr(ctx, "request_context"):
            req_ctx = ctx.request_context
            if hasattr(req_ctx, "state"):
                return getattr(req_ctx.state, "principal", None)
            if hasattr(req_ctx, "principal"):
                return req_ctx.principal
            if isinstance(req_ctx, dict):
                return req_ctx.get("principal")
        return None

    def format_tool_response(
        self,
        tool_name: str,
        result: Any,
        *,
        as_xml: bool = False,
    ) -> str:
        """Format a tool response for LLM consumption.

        Parameters
        ----------
        tool_name : str
            Name of the tool that produced the result.
        result : Any
            The tool's return value.
        as_xml : bool
            If True, wrap in XML envelope format.

        Returns
        -------
        str
            Formatted response string.
        """
        if as_xml:
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, indent=2, default=str)

            return (
                f'<tool_response tool_name="{tool_name}">\n'
                f"  <llm_output><![CDATA[\n{content}\n  ]]></llm_output>\n"
                f"</tool_response>"
            )

        # Default JSON format
        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2, default=str)

    def format_error_response(
        self,
        tool_name: str,
        error_code: str,
        detail: str,
        *,
        retryable: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Format an error response for LLM consumption.

        Parameters
        ----------
        tool_name : str
            Name of the tool that failed.
        error_code : str
            Short error code (e.g., "VALIDATION_ERROR").
        detail : str
            Human-readable error description.
        retryable : bool
            Whether the operation might succeed on retry.
        meta : dict | None
            Additional error metadata.

        Returns
        -------
        str
            Formatted error response string.
        """
        meta_block = ""
        if meta:
            try:
                meta_json = json.dumps(meta, separators=(",", ":"))
                meta_block = f"\n    <meta>{meta_json}</meta>"
            except Exception:
                pass

        return (
            f'<tool_response tool_name="{tool_name}">\n'
            f'  <error code="{error_code}" retryable="{"true" if retryable else "false"}">\n'
            f"    <![CDATA[{detail}]]>{meta_block}\n"
            f"  </error>\n"
            f"</tool_response>"
        )

    def get_tool_annotations(self, tool_name: str) -> dict[str, Any]:
        """Get the annotations for a tool (server_populated, depends_on).

        Parameters
        ----------
        tool_name : str
            Name of the tool.

        Returns
        -------
        dict
            Annotations dictionary with "server_populated" and "depends_on" keys.
        """
        meta = self._tool_metadata.get(tool_name)
        if meta is None:
            return {}
        return {
            "server_populated": meta.server_populated,
            "depends_on": meta.depends_on,
        }

