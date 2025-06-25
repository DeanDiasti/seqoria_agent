from __future__ import annotations

"""Public import location for :class:`LocalToolbox`."""

from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from seqoria_agent.server.xmcp_core import AuthBridge, ToolCore
from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend
from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.xmcp_core.models import ToolInfo as _ToolInfo  # noqa: F401 – re-export

__all__: list[str] = ["LocalToolbox"]


class LocalToolbox:  # noqa: D101 – public façade
    def __init__(
        self,
        *,
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        auth_backend: BaseAuthBackend | None = None,
    ) -> None:
        if auth_backend is None:
            auth_backend = NoAuthBackend()
        if not isinstance(auth_backend, BaseAuthBackend):  # pragma: no cover
            raise TypeError("auth_backend must derive from BaseAuthBackend")
        self._auth_backend = auth_backend

        # Shared execution engine ---------------------------------------
        self._core = ToolCore(
            strict_mode=strict_mode,
            strict_server_params=strict_server_params,
            json_wrapper=json_wrapper,
        )
        self.tool = self._core.tool  # type: ignore[attr-defined]
        self.display_tool = self._core.display_tool  # type: ignore[attr-defined]

        # Expose internals (legacy)
        self._tools: Dict[str, _ToolInfo] = self._core.tools  # type: ignore[assignment]
        self._tool_formatters = self._core._tool_formatters  # noqa: SLF001
        self._param_providers: List[Callable] = self._core._param_providers  # noqa: SLF001

        # Authentication bridge ----------------------------------------
        self._auth_bridge = AuthBridge(self._auth_backend)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        *,
        server_args: Dict[str, Any] | None = None,
    ) -> List[SimpleNamespace]:  # noqa: D401
        if name not in self._tools:
            raise XMCPError(code="UNKNOWN_TOOL", detail=name, retryable=False)

        arguments = dict(arguments or {})
        server_args = server_args or {}
        info = self._tools[name]

        # Fabricate minimal request stub --------------------------------
        fake_req = SimpleNamespace(state=SimpleNamespace())
        await self._auth_bridge.authenticate(fake_req)
        self._auth_bridge.ensure_principal_or_error(name)

        # Parameter injection (server-populated) ------------------------
        if info.server_params or server_args:
            arguments = await self._core.inject_server_params(
                request_or_ns=fake_req,
                tool_name=name,
                info=info,
                arguments=arguments,
                server_args=server_args,
            )

        xml = await self._core.run_tool(name, arguments)
        return [SimpleNamespace(text=xml)]

    # ------------------------------------------------------------------
    # Introspection helper used by AIClient
    # ------------------------------------------------------------------

    def list_tools(self):  # noqa: D401 – mimic MCP list_tools
        items: List[Any] = []
        for name, info in self._tools.items():
            ann = {"is_display_tool": info.display}
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
    # Parameter provider registration
    # ------------------------------------------------------------------

    def register_param_provider(self, fn: Callable):  # noqa: D401
        self._param_providers.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Tests may access this helper – delegate to core
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, *a, **kw):  # noqa: D401, ANN001
        return self._core._wrap_tool_with_error_handling(*a, **kw) 