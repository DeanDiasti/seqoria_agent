from __future__ import annotations

# pragma: no cover
"""Authentication bridge – framework-agnostic helper shared by FastXMCP & LocalToolbox.

The :class:`AuthBridge` centralises *once-per-request* authentication so the
public façades no longer duplicate the same boiler-plate.  It purposefully
avoids importing *FastAPI* or *Starlette* thereby remaining usable in any
runtime that provides a *request-like* object (duck-typed).
"""

# from types import SimpleNamespace  # removed – keep import minimal
from typing import Any
import inspect

from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend
from seqoria_agent.models.errors import XMCPError

__all__: list[str] = [
    "AuthBridge",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Sentinel value so we can differentiate *missing* attribute from "set to
# None" (unauthenticated).
class _Unset:  # noqa: D401 – internal marker
    pass


_UNSET = _Unset()


# ---------------------------------------------------------------------------
# Public mix-in
# ---------------------------------------------------------------------------


class AuthBridge:  # noqa: D401 – lightweight utility class
    """Authenticate a request exactly *once* and expose the *principal*.

    The bridge is agnostic to the concrete HTTP framework.  It expects a
    *request-like* object that either:

    1. carries a ``.state`` attribute (Starlette / FastAPI) or
    2. allows arbitrary attributes (e.g. :class:`types.SimpleNamespace`).

    Both variants are handled transparently so the same logic can operate on
    real HTTP requests *and* the stub objects fabricated by
    :class:`~seqoria_agent.toolbox.LocalToolbox`.
    """

    def __init__(self, backend: BaseAuthBackend):
        if not isinstance(backend, BaseAuthBackend):
            raise TypeError("backend must be an instance of BaseAuthBackend")
        self._backend: BaseAuthBackend = backend
        # We remember the *last* request object we authenticated so that
        # :pymeth:`ensure_principal_or_error` can examine the resolved
        # principal without having to pass the request around explicitly.
        # This is a pragmatic compromise – the public façades (FastXMCP &
        # LocalToolbox) call the two methods sequentially *within* the same
        # request scope, therefore the reference is safe.
        self._last_request: Any | None = None

    # ------------------------------------------------------------------
    async def authenticate(self, request_like: Any):  # noqa: D401, ANN001
        """Run *backend* at most once and store ``principal`` on *request_like*.

        The function first checks whether a ``principal`` is **already** set on
        the incoming object.  When present we return it immediately so that a
        *middleware*-based backend (which executed earlier in the request
        lifecycle) is **not** invoked a second time.  Otherwise the underlying
        backend's :py:meth:`authenticate` coroutine is executed and its return
        value stored on the request for later reuse.
        """
        principal = self._get_principal(request_like, default=_UNSET)
        if principal is not _UNSET:
            # Backend has already run (middleware or previous explicit call).
            self._last_request = request_like
            return principal if principal is not None else None

        # Run backend (sync- or async-compatible)
        res = self._backend.authenticate(request_like)
        if inspect.isawaitable(res):  # type: ignore[arg-type]
            res = await res  # type: ignore[assignment]

        # Always expose attribute so downstream code can tell unauthenticated
        self._set_principal(request_like, res)
        # Remember for later checks
        self._last_request = request_like
        return res  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def ensure_principal_or_error(self, tool_name: str):  # noqa: D401
        """Raise *UNAUTHENTICATED* error when backend *requires* auth but none.

        A backend that derives from :class:`NoAuthBackend` explicitly *does not*
        require authentication, therefore the check is skipped.  For all other
        backends we demand that the previously stored ``principal`` is *truthy*
        (not ``None``) and raise :class:`XMCPError` otherwise.
        """
        if isinstance(self._backend, NoAuthBackend):
            return  # unauthenticated allowed

        if self._last_request is None:
            # Programmer error – authenticate() must be called first.
            raise XMCPError(code="UNAUTHENTICATED", detail=tool_name, retryable=False)

        principal = self._get_principal(self._last_request, default=None)
        if principal is None:
            raise XMCPError(code="UNAUTHENTICATED", detail=tool_name, retryable=False)

    # ------------------------------------------------------------------
    # Helper – attribute handling
    # ------------------------------------------------------------------

    @staticmethod
    def _get_principal(obj: Any, *, default: Any = None):  # noqa: ANN001
        if hasattr(obj, "state"):
            return getattr(obj.state, "principal", default)
        return getattr(obj, "principal", default)

    @staticmethod
    def _set_principal(obj: Any, value: Any):  # noqa: ANN001
        if hasattr(obj, "state"):
            obj.state.principal = value
        else:
            setattr(obj, "principal", value) 