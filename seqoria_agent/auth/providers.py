from __future__ import annotations

"""Client-side authentication provider abstractions (Phase-0 scaffolding).

The provider concept will be fully integrated in Phase-2.  For now we expose
minimal base classes so that application code can already import them without
causing import errors.
"""

from abc import ABC, abstractmethod
from typing import Dict

__all__: list[str] = [
    "BaseAuthProvider",
    "NoAuthProvider",
    "StaticCookieProvider",
    "StaticBearerTokenProvider",
    "StaticApiKeyProvider",
]


class BaseAuthProvider(ABC):  # noqa: D101 – small interface
    """Produce HTTP headers/cookies for outbound requests (client-side)."""

    @abstractmethod
    async def auth_headers(self) -> Dict[str, str]:  # noqa: D401
        """Return mapping merged into every HTTP request performed by AIClient."""

    # ------------------------------------------------------------------
    # Optional token-refresh hook (Phase-4) -----------------------------
    # ------------------------------------------------------------------

    async def refresh_token_if_needed(self) -> None:  # noqa: D401
        """Default no-op refresh hook called before each request.

        Providers that need to refresh an access token (e.g. OAuth2 implicit
        flow with expiry) can override this coroutine.  The function must be
        *idempotent* and *fast* because it is executed on **every** outbound
        call.
        """
        return None


class NoAuthProvider(BaseAuthProvider):  # noqa: D101 – trivial default
    """Provider that adds **no** authentication headers (Phase 0 default)."""

    async def auth_headers(self) -> Dict[str, str]:  # noqa: D401
        return {}


# ------------------------------------------------------------------
# Built-in static helpers (Phase-3) ---------------------------------
# ------------------------------------------------------------------

class StaticCookieProvider(NoAuthProvider):  # noqa: D101 – simple header helper
    """Attach a static cookie containing the JWT to every request."""

    def __init__(self, token: str, *, cookie_name: str = "access_token") -> None:
        self._cookie = f"{cookie_name}={token}"

    async def auth_headers(self) -> Dict[str, str]:  # noqa: D401
        # HTTP headers are case-insensitive; use canonical 'cookie'
        return {"cookie": self._cookie}


class StaticBearerTokenProvider(NoAuthProvider):  # noqa: D101 – Authorization header helper
    """Attach a static ``Authorization: Bearer <token>`` header."""

    def __init__(self, token: str):
        self._hdr_val = f"Bearer {token}"

    async def auth_headers(self) -> Dict[str, str]:  # noqa: D401
        return {"authorization": self._hdr_val}


# ------------------------------------------------------------------
# NEW – Static API-key header provider (Phase-5)
# ------------------------------------------------------------------


class StaticApiKeyProvider(NoAuthProvider):  # noqa: D101 – static API-key header helper
    """Attach a static *API-key* header to every request.

    Parameters
    ----------
    token:
        The API-key to send.
    header_name:
        Name of the HTTP header that carries the token (default "X-API-Key").
    """

    def __init__(self, token: str, *, header_name: str = "X-API-Key") -> None:
        self._header_name = header_name
        self._token = token

    async def auth_headers(self) -> Dict[str, str]:  # noqa: D401
        return {self._header_name: self._token} 