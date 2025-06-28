from __future__ import annotations

"""Thin FastXMCPRouter façade (public import location)."""

from typing import Any, Callable, Dict, List

from fastapi import APIRouter

from seqoria_agent.server.xmcp_core import ToolCore
from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend
from seqoria_agent.models.errors import XMCPError

# Re-export legacy helper for type annotations
from seqoria_agent.server.xmcp_core.models import ToolInfo as _ToolInfo  # noqa: F401

__all__: list[str] = ["FastXMCPRouter"]


class FastXMCPRouter(APIRouter):  # noqa: D101
    def __init__(
        self,
        *,
        prefix: str | None = "",
        tags: List[str] | None = None,
        strict_mode: bool = False,
        strict_server_params: bool = True,
        json_wrapper: bool = False,
        auth_backend: BaseAuthBackend | None = None,
        **kw,
    ) -> None:
        if auth_backend is None:
            auth_backend = NoAuthBackend()
        if not isinstance(auth_backend, BaseAuthBackend):  # pragma: no cover
            raise TypeError("auth_backend must derive from BaseAuthBackend")

        super().__init__(prefix=prefix or "", tags=tags or [], **kw)

        self._auth_backend = auth_backend

        # Dedicated ToolCore instance ------------------------------------
        self._core = ToolCore(
            strict_mode=strict_mode,
            strict_server_params=strict_server_params,
            json_wrapper=json_wrapper,
        )
        self.tool = self._core.tool  # type: ignore[attr-defined]
        self.display_tool = self._core.display_tool  # type: ignore[attr-defined]

        # Expose internals so parent app can merge -----------------------
        self._tools: Dict[str, _ToolInfo] = self._core.tools  # type: ignore[assignment]
        self._tool_formatters = self._core._tool_formatters  # noqa: SLF001
        self._param_providers: List[Callable] = self._core._param_providers  # noqa: SLF001

    # ------------------------------------------------------------------
    # Public helper – parameter provider registration
    # ------------------------------------------------------------------

    def register_param_provider(self, fn: Callable):  # noqa: D401
        self._param_providers.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Tests may access this for strict-mode checks – delegate to core
    # ------------------------------------------------------------------

    def _wrap_tool_with_error_handling(self, *a, **kw):  # noqa: D401, ANN001
        return self._core._wrap_tool_with_error_handling(*a, **kw) 