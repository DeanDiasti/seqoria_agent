from __future__ import annotations

"""Server-side authentication backend abstractions (Phase-0 scaffolding).

The concrete authentication logic will be implemented in later phases.  For
now we only provide a *no-operation* backend so that existing applications
continue to work unchanged when they **do not** opt-in to authentication.

See `oauth-implementation.md` for the full design overview.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

__all__: list[str] = [
    "BaseAuthBackend",
    "NoAuthBackend",
    "BearerCookieBackend",
    "HeaderApiKeyBackend",
    "BaseMiddlewareAuthBackend",
]


# ------------------------------------------------------------------
# Core abstractions --------------------------------------------------
# ------------------------------------------------------------------


class BaseAuthBackend(ABC):  # noqa: D101 – succinct interface
    """Abstract authentication backend used by FastXMCP & LocalToolbox.

    Subclasses must implement :meth:`authenticate` and either return a
    *principal* object (any Python object that represents the authenticated
    user/client) or raise an exception (usually *fastapi.HTTPException*) to
    abort the request.  Implementations **must not** return *None* to signal
    failure – raise instead so the correct HTTP status code is propagated.
    """

    @abstractmethod
    async def authenticate(self, request) -> Any:  # noqa: ANN401 – request is duck-typed
        """Validate *request* and return a principal on success.

        Implementations may be *sync* or *async*.  For convenience, the base
        class provides :pyfunc:`run` which handles both transparently.
        """

    # ------------------------------------------------------------------
    # Convenience helper – handle sync/async subclasses uniformly
    # ------------------------------------------------------------------

    async def _maybe_call(self, request):  # noqa: D401, ANN001
        res = self.authenticate(request)
        if inspect.isawaitable(res):  # type: ignore[arg-type]
            res = await res  # type: ignore[assignment]
        return res

    # ------------------------------------------------------------------
    # FastAPI integration helper (Phase-4) ------------------------------
    # ------------------------------------------------------------------

    def as_dependency(self):  # noqa: D401
        """Return a FastAPI *Depends* object executing this backend once and
        providing the *principal* to route handlers.  This is a convenience
        wrapper so users can reuse the same backend logic for classic REST
        endpoints alongside XMCP tools.
        """
        try:
            from fastapi import Depends, Request  # local import, optional
            globals().setdefault("Request", Request)  # make available for postponed annotations
        except Exception:  # pragma: no cover – FastAPI unavailable outside server runtime
            raise RuntimeError("fastapi is required for 'as_dependency()' usage")

        async def _dep(request: Request):  # type: ignore[override]
            return await self._maybe_call(request)

        return Depends(_dep)


class NoAuthBackend(BaseAuthBackend):  # noqa: D101 – intentional stub
    """Default backend that performs **no authentication** (Phase 0 default)."""

    async def authenticate(self, request):  # noqa: ANN001 – duck-typed request
        # Simply return *None* so downstream logic can detect unauthenticated
        # scenarios when needed.  This keeps behaviour identical to the pre-auth
        # implementation where no principal information existed.
        return None 


# ------------------------------------------------------------------
# NEW – Middleware-ready mixin ---------------------------------------
# ------------------------------------------------------------------

try:
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – FastAPI optional at runtime
    BaseHTTPMiddleware = object  # type: ignore
    JSONResponse = None  # type: ignore


class BaseMiddlewareAuthBackend(BaseHTTPMiddleware, BaseAuthBackend):  # type: ignore[misc]
    """Auth backend variant that is also a Starlette *HTTP middleware*.

    Sub-classes inherit the regular :meth:`authenticate` contract and can be
    mounted via ``app.add_middleware(SubBackend, **cfg)`` on any FastAPI app.
    The implementation ensures the authenticated *principal* is stored on
    ``request.state.principal`` **once per request** before routing proceeds.
    """

    async def dispatch(self, request, call_next):  # noqa: ANN001
        try:
            principal = self.authenticate(request)
            if inspect.isawaitable(principal):  # type: ignore[arg-type]
                principal = await principal  # type: ignore[assignment]
            # Always expose attribute so downstream code can tell unauthenticated
            request.state.principal = principal
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001 – bubble up 401 etc. cleanly
            try:
                from fastapi import HTTPException as _HTTPExc  # local import
            except Exception:  # pragma: no cover – FastAPI missing
                raise

            if isinstance(exc, _HTTPExc) and JSONResponse is not None:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)  # type: ignore[call-arg]
            raise


# ------------------------------------------------------------------
# Concrete backends (now middleware-capable) -------------------------
# ------------------------------------------------------------------


class BearerCookieBackend(BaseMiddlewareAuthBackend):  # noqa: D101 – JWT-in-cookie backend
    """Validate JWT carried in a cookie and return decoded claims dict.

    Parameters
    ----------
    secret:
        Shared secret used to verify the token signature.
    algorithm:
        Hashing algorithm (e.g. "HS256").  Passed directly to ``jwt.decode``.
    cookie_name:
        Name of the cookie containing the token (default: "access_token").
    """

    def __init__(self, *args, secret: str | None = None, algorithm: str = "HS256", cookie_name: str = "access_token", **kw) -> None:  # noqa: D401, ANN001
        """Create backend.

        The *first positional* argument is optionally the ASGI *app* when the
        backend is mounted via ``add_middleware``.  All remaining parameters
        are identical to the pre-middleware constructor so existing code keeps
        working unchanged.
        """

        app = None
        if args and not isinstance(args[0], str):
            app, *args = args  # noqa: PLW2901 – rebind for clarity

        # Backwards compatibility: allow ``BearerCookieBackend(secret)`` style
        if secret is None and args:
            secret, *args = args  # type: ignore[assignment]

        if secret is None:
            raise TypeError("'secret' argument is required")

        # Initialise BaseHTTPMiddleware only when an *app* is supplied (i.e.,
        # middleware mode).  In plain backend mode this would be unnecessary
        # and could break unit-tests that fabricate the backend without FastAPI.
        if app is not None and BaseHTTPMiddleware is not object:
            BaseHTTPMiddleware.__init__(self, app)  # type: ignore[misc]

        try:
            import jwt as _jwt  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover – missing optional dep
            raise RuntimeError("'pyjwt' package is required for BearerCookieBackend") from exc

        self._jwt = _jwt
        self._secret = secret
        self._algo = algorithm
        self._cookie_name = cookie_name
        # Expose config for FastAPI.add_middleware(**kwargs)
        self.secret = secret
        self.algorithm = algorithm
        self.cookie_name = cookie_name

    async def authenticate(self, request):  # noqa: ANN001 – FastAPI/Starlette request duck-typed
        from fastapi import HTTPException  # type: ignore – only imported inside server context

        token = request.cookies.get(self._cookie_name) if hasattr(request, "cookies") else None
        if not token:
            raise HTTPException(status_code=401, detail="Missing authentication cookie")
        try:
            claims = self._jwt.decode(token, self._secret, algorithms=[self._algo])
        except self._jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except self._jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Use claims as principal; callers may wrap this further.
        return claims 


# ------------------------------------------------------------------
# NEW – Simple API-Key header backend (Phase-5)
# ------------------------------------------------------------------


class HeaderApiKeyBackend(BaseMiddlewareAuthBackend):  # noqa: D101 – static header backend
    """Validate an API-key supplied via custom HTTP header.

    Parameters
    ----------
    key_header:
        Name of the HTTP header carrying the API-key.  Matching is case-insensitive.
    validator:
        Either a *callable* ``fn(token:str) -> bool`` that returns *True* when
        the token is accepted, or a *str* containing the **exact** allowed
        token value.  When *None* (default) any *non-empty* token is accepted.
    principal_factory:
        Optional callable ``fn(token:str) -> Any`` to convert the raw token
        into a *principal* object exposed to tools via ``request.state``.
        Defaults to returning ``{"api_key": token}``.
    """

    def __init__(
        self,
        *args,
        key_header: str = "X-API-Key",
        validator: "Callable[[str], bool] | str | None" = None,
        principal_factory: "Callable[[str], Any] | None" = None,
        **kw,
    ) -> None:
        """Create backend.

        The optional first positional *args* may carry the ASGI *app* when the
        backend is mounted as middleware.  All keyword parameters preserve the
        legacy constructor.
        """

        app = None
        if args and not isinstance(args[0], str):
            app, *args = args  # noqa: PLW2901

        if app is not None and BaseHTTPMiddleware is not object:
            BaseHTTPMiddleware.__init__(self, app)  # type: ignore[misc]

        from inspect import isfunction  # local import to avoid heavy deps

        self._header = key_header

        if validator is None:
            self._validator = lambda t: bool(t)
        elif isinstance(validator, str):
            self._validator = lambda t, _v=validator: t == _v  # noqa: E731 – small lambda
        elif isfunction(validator) or callable(validator):
            self._validator = validator  # type: ignore[assignment]
        else:
            raise TypeError("validator must be a str, callable or None")

        self._make_principal = (
            principal_factory if principal_factory is not None else (lambda tok: {"api_key": tok})
        )

        # Expose config for FastAPI.add_middleware(**kwargs)
        self.key_header = key_header
        self.validator = validator
        self.principal_factory = principal_factory

    async def authenticate(self, request):  # noqa: ANN001 – FastAPI/Starlette request duck-typed
        from fastapi import HTTPException  # type: ignore – only imported inside server context

        token = request.headers.get(self._header)
        if not token or not self._validator(token):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return self._make_principal(token) 