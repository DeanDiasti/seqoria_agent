from __future__ import annotations
# pragma: no cover

"""Server-populated parameter injection helper (Phase-2).

This module centralises the logic that resolves *server-populated parameters*
(SPP) requested by an XMCP tool.  It is deliberately **framework-agnostic** –
no FastAPI or Starlette imports – and therefore can be reused by the HTTP
runtime (``FastXMCP``), the APIRouter variant as well as the in-process
``LocalToolbox``.

Responsibilities
----------------
* Detect when the caller/LLM tries to *override* a protected SPP and – when
  *strict* mode is enabled – raise a structured ``XMCPError``.
* Inject built-in values for ``principal`` and ``user_id`` based on the
  request-local principal object provided by the active authentication
  backend.
* Delegate remaining parameters to *provider* callables registered by the
  application.  The first provider that returns a non-``None`` value wins.
* Emit a ``MISSING_SERVER_PARAM`` error when a required SPP cannot be
  satisfied.

The implementation has **zero external dependencies** besides the core
``seqoria_agent`` models.
"""

from typing import Any, Callable, Dict, Sequence
import inspect

from seqoria_agent.models.errors import XMCPError

# Local import – avoids circular dependency for typing only
from .models import ToolInfo  # pylint: disable=cyclic-import  # noqa: F401

__all__: list[str] = ["ParamInjector"]


class ParamInjector:
    """Resolve server-populated parameters for a *single* tool invocation."""

    def __init__(self, *, providers: Sequence[Callable], strict_mode: bool) -> None:  # noqa: D401
        # Keep original *list* reference so modifications via ToolCore are
        # visible here without additional bookkeeping.
        self._providers: list[Callable] = list(providers) if not isinstance(providers, list) else providers
        self._strict = bool(strict_mode)

    async def inject(self,
                     request_or_ns: Any,
                     tool_name: str,
                     info: ToolInfo,
                     *,
                     arguments: dict[str, Any] | None = None,
                     server_args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a new *arguments* mapping with SPP injected.

        Parameters
        ----------
        request_or_ns : Any
            Either a real Starlette/FastAPI ``Request`` *or* the fabricated
            ``SimpleNamespace`` used by :class:`~seqoria_agent.toolbox.LocalToolbox`.
        tool_name : str
            Name of the tool - only required for informative error messages.
        info : ToolInfo
            Registry entry describing the tool.
        arguments : dict, optional
            Arguments provided by the caller/LLM (``tools/call`` → ``arguments``).
        server_args : dict, optional
            Explicit ``serverArgs`` override supplied by the caller.

        Returns
        -------
        dict
            Fully-populated argument mapping suitable for invoking the tool
            callable.
        """
        arguments = dict(arguments or {})  # shallow copy – we will mutate
        server_args = dict(server_args or {})

        if self._strict:
            for protected in info.server_params:
                if protected in arguments:
                    raise XMCPError(code="FORBIDDEN_PARAM_OVERRIDE", detail=protected, retryable=False)

        for key in server_args:
            if key not in info.server_params:
                raise XMCPError(code="UNKNOWN_SERVER_PARAM", detail=key, retryable=False)
            if key in arguments and self._strict:
                raise XMCPError(code="FORBIDDEN_PARAM_OVERRIDE", detail=key, retryable=False)

        # Merge explicit server_args – they take precedence over arguments
        if server_args:
            arguments.update(server_args)

        principal_obj = None
        if hasattr(request_or_ns, "state"):
            principal_obj = getattr(request_or_ns.state, "principal", None)
        # Fallback for SimpleNamespace where user may attach principal directly
        if principal_obj is None:
            principal_obj = getattr(request_or_ns, "principal", None)

        if "principal" in info.server_params and "principal" not in arguments:
            arguments["principal"] = principal_obj

        if "user_id" in info.server_params and "user_id" not in arguments:
            if principal_obj is not None:
                if hasattr(principal_obj, "id"):
                    arguments["user_id"] = getattr(principal_obj, "id")
                elif isinstance(principal_obj, dict) and "id" in principal_obj:
                    arguments["user_id"] = principal_obj["id"]

        for param in info.server_params:
            if param in arguments:
                continue  # already resolved

            value_found = False
            for provider in self._providers:
                try:
                    res = provider(request_or_ns, tool_name, param)
                    if inspect.isawaitable(object=res):
                        res = await res  # type: ignore[assignment]
                except Exception:  # pragma: no cover – provider bug
                    continue  # skip faulty provider

                if res is not None:
                    arguments[param] = res
                    value_found = True
                    break

            if not value_found:
                # Missing even after trying every provider – error
                raise XMCPError(code="MISSING_SERVER_PARAM", detail=param, retryable=False)

        return arguments 