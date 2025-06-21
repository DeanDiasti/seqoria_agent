from __future__ import annotations

"""Parameter provider abstraction for injecting server-populated parameters on the client side.

A *ParameterProvider* is a small object (or function with a ``resolve``
method) that can supply values for parameters the LLM is not allowed to
know or set.  AIClient will iterate through the list of providers until a
non-None value is returned.
"""

from typing import Any, Protocol, TypedDict, Optional, List

__all__: list[str] = [
    "ParamContext",
    "ParameterProvider",
]


class ParamContext(TypedDict, total=False):
    """Context data passed to providers.

    It intentionally stays open-ended – applications can inject whatever
    extra metadata they need (e.g. IP address, auth claims, request ID).
    """

    tenant_id: str
    history: list[dict]
    request_meta: dict


class ParameterProvider(Protocol):
    """Protocol for client-side parameter providers."""

    async def resolve(
        self,
        param_name: str,
        tool_name: str,
        context: ParamContext,
    ) -> Optional[Any]: ... 