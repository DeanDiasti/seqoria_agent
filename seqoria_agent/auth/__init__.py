"""Authentication scaffolding (Phase 0).

This package introduces minimal no-op defaults used by the initial
implementation phases.  The real logic will follow in subsequent
commits (see *oauth-implementation.md*).
"""

# Re-export public interface --------------------------------------------------
from .backends import BaseAuthBackend, NoAuthBackend, BearerCookieBackend, HeaderApiKeyBackend, BaseMiddlewareAuthBackend  # noqa: F401
from .providers import (
    BaseAuthProvider,
    NoAuthProvider,
    StaticCookieProvider,
    StaticBearerTokenProvider,
    StaticApiKeyProvider,
)  # noqa: F401

__all__: list[str] = [
    "BaseAuthBackend",
    "NoAuthBackend",
    "BearerCookieBackend",
    "HeaderApiKeyBackend",
    "BaseMiddlewareAuthBackend",
    "BaseAuthProvider",
    "NoAuthProvider",
    "StaticCookieProvider",
    "StaticBearerTokenProvider",
    "StaticApiKeyProvider",
] 