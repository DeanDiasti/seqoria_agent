"""LocalToolbox - In-process tool execution using MCP primitives.

This module provides LocalToolbox, which allows tools to be registered and
executed in-process without an HTTP server. Pydantic validates arguments and
generates tool schemas; tools execute directly in Python.

Example
-------
>>> from seqoria_agent.toolbox import LocalToolbox
>>>
>>> toolbox = LocalToolbox()
>>>
>>> @toolbox.tool(server_populated=["user_id"])
>>> async def greet(name: str, user_id: str) -> str:
...     return f"Hello {name}! (user: {user_id})"
>>>
>>> result = await toolbox.call_tool("greet", {"name": "Alice"}, server_args={"user_id": "123"})
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

__all__ = ["LocalToolbox", "ToolInfo"]


@dataclass
class ToolInfo:
    """Metadata for a registered tool.

    Attributes
    ----------
    func : Callable
        The wrapped tool function.
    description : str | None
        Tool description for LLM.
    server_params : list[str]
        Server-populated parameter names.
    depends_on : list[str]
        Tool dependencies.
    input_schema : dict
        JSON Schema for input parameters.
    """

    func: Callable[..., Any]
    description: str | None = None
    server_params: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    argument_model: type[BaseModel] | None = None


class LocalToolbox:
    """In-process tool registry and executor.

    Provides tool registration and execution without requiring an HTTP server.
    Tools can have server-populated parameters and dependencies, similar to
    remote MCP servers.

    Parameters
    ----------
    strict_server_params : bool
        If True, raise error when caller provides server-populated params.

    Example
    -------
    >>> toolbox = LocalToolbox()
    >>>
    >>> @toolbox.tool()
    >>> def add(a: int, b: int) -> int:
    ...     return a + b
    >>>
    >>> result = await toolbox.call_tool("add", {"a": 1, "b": 2})
    """

    def __init__(
        self,
        *,
        strict_server_params: bool = True,
    ) -> None:
        self._strict_server_params = strict_server_params

        # Tool registry
        self._tools: Dict[str, ToolInfo] = {}

        # Parameter providers for server-populated params
        self._param_providers: List[Callable[..., Any]] = []

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        server_populated: Sequence[str] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a tool with the toolbox.

        Parameters
        ----------
        name : str | None
            Tool name (defaults to function name).
        description : str | None
            Tool description (defaults to docstring).
        server_populated : Sequence[str] | None
            Parameter names that are server-populated.
        depends_on : Sequence[str] | None
            Names of tools that must execute before this one.

        Returns
        -------
        Callable
            Decorator function.

        Example
        -------
        >>> @toolbox.tool(server_populated=["user_id"])
        >>> async def my_tool(name: str, user_id: str) -> str:
        ...     return f"Hello {name} (user: {user_id})"
        """
        server_populated_list = list(server_populated or [])
        depends_on_list = list(depends_on or [])

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]

            # Generate input schema
            argument_model = self._create_argument_model(fn, server_populated_list)
            input_schema = argument_model.model_json_schema()

            # Wrap function for async execution
            @functools.wraps(fn)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                if inspect.iscoroutinefunction(fn):
                    result = fn(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(fn, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result

            # Register tool
            self._tools[tool_name] = ToolInfo(
                func=wrapped,
                description=tool_desc,
                server_params=server_populated_list,
                depends_on=depends_on_list,
                input_schema=input_schema,
                argument_model=argument_model,
            )

            return fn

        return decorator

    def _create_argument_model(
        self,
        fn: Callable[..., Any],
        server_populated: list[str],
    ) -> type[BaseModel]:
        """Build the public argument schema and validator once at registration."""
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)
        unknown = set(server_populated) - set(sig.parameters)
        if unknown:
            raise ValueError(f"Unknown server-populated parameters: {', '.join(sorted(unknown))}")
        fields = {}
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError("Tool parameters must accept keyword arguments")
            if name in server_populated or name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError("Tool parameters must be explicitly named")
            annotation = hints.get(name, Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)
        return create_model(
            f"{fn.__name__}Arguments", __config__=ConfigDict(extra="forbid"), **fields
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        server_args: dict[str, Any] | None = None,
    ) -> List[SimpleNamespace]:
        """Execute a tool by name.

        Parameters
        ----------
        name : str
            Tool name to execute.
        arguments : dict | None
            Arguments from the caller/LLM.
        server_args : dict | None
            Server-populated arguments.

        Returns
        -------
        List[SimpleNamespace]
            List containing result wrapped in SimpleNamespace with .text attribute.

        Raises
        ------
        ValueError
            If tool is not found.
        RuntimeError
            If server-populated params are missing.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in LocalToolbox")

        info = self._tools[name]
        arguments = dict(arguments or {})
        server_args = dict(server_args or {})

        # Validate server_args
        for key in server_args:
            if key not in info.server_params:
                raise ValueError(f"Unknown server parameter: {key}")

        # Check for forbidden overrides
        if self._strict_server_params:
            for param in info.server_params:
                if param in arguments:
                    raise ValueError(f"Cannot override server-populated param: {param}")

        # Merge server_args into arguments
        arguments.update(server_args)

        # Resolve missing server-populated params via providers
        for param in info.server_params:
            if param not in arguments:
                value = await self._resolve_param(name, param)
                if value is not None:
                    arguments[param] = value

        # Check all server-populated params are resolved
        missing = [p for p in info.server_params if p not in arguments]
        if missing:
            raise RuntimeError(
                f"Missing server-populated parameters for '{name}': {', '.join(missing)}"
            )

        # Validate public arguments with the same model used for discovery.
        # Attribute access preserves nested Pydantic instances for typed tools.
        if info.argument_model is not None:
            hidden = {key: value for key, value in arguments.items() if key in info.server_params}
            public = {
                key: value for key, value in arguments.items() if key not in info.server_params
            }
            validated = info.argument_model.model_validate(public)
            arguments = {key: getattr(validated, key) for key in type(validated).model_fields}
            arguments.update(hidden)
        result = await info.func(**arguments)
        text = self._format_result(name, result)

        return [SimpleNamespace(text=text)]

    async def _resolve_param(self, tool_name: str, param_name: str) -> Any | None:
        """Resolve a server-populated parameter via providers."""
        for provider in self._param_providers:
            try:
                # Call provider with (request, tool_name, param_name) signature
                # For local toolbox, we pass None as request
                if inspect.iscoroutinefunction(provider):
                    result = provider(None, tool_name, param_name)
                else:
                    result = await asyncio.to_thread(provider, None, tool_name, param_name)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result as string."""
        if isinstance(result, str):
            content = result
        else:
            content = json.dumps(result, indent=2, default=str)
        return content

    def _format_error(self, tool_name: str, exc: Exception) -> str:
        """Format tool error as string."""
        return f"Error in {tool_name}: {exc}"

    def list_tools(self) -> List[SimpleNamespace]:
        """List all registered tools with their metadata.

        Returns
        -------
        List[SimpleNamespace]
            List of tool info objects with name, description, inputSchema, annotations.
        """
        items: List[SimpleNamespace] = []
        for name, info in self._tools.items():
            annotations: dict[str, Any] = {}
            if info.server_params:
                annotations["server_populated"] = info.server_params
            if info.depends_on:
                annotations["depends_on"] = info.depends_on

            items.append(
                SimpleNamespace(
                    name=name,
                    description=info.description,
                    inputSchema=info.input_schema,
                    annotations=annotations,
                )
            )
        return items

    def register_param_provider(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a parameter provider for server-populated parameters.

        The provider is called with (request, tool_name, param_name) and should
        return the value or None if it cannot provide the parameter.

        Parameters
        ----------
        fn : Callable
            Provider function.

        Returns
        -------
        Callable
            The same function (allows use as decorator).
        """
        self._param_providers.append(fn)
        return fn

    def get_tool_info(self, name: str) -> ToolInfo | None:
        """Get metadata for a specific tool."""
        return self._tools.get(name)
