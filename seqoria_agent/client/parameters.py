from __future__ import annotations

"""Parameter resolution utilities for *Server-Populated Parameters* (SPP)."""

from typing import Any, Awaitable, Callable, Mapping, Optional, Union, Type
import inspect
import asyncio
import logging
from pydantic import BaseModel

__all__: list[str] = ["ParamContext",
                      "ParameterSource",
                      "ParameterRegistry"]

ParamContext = Mapping[str, Any]

ParameterSource = Union[Any, Callable[..., Awaitable[Any] | Any]]

class ParameterRegistry:  # noqa: D101 – concise implementation
    """Resolve server-populated parameters from a mapping.

    Examples
    --------
    async def demo():
        async def get_user_id(tenant_id: str | None = None) -> str:
            return "u-42" if tenant_id == "acme" else "anonymous"
   
        reg = ParameterRegistry({
            "tenant_id": "acme",
            "user_id": get_user_id,
        })
        ctx: ParamContext = {"tenant_id": "acme"}
        assert await reg.resolve("tenant_id", context=ctx) == "acme"
        assert await reg.resolve("user_id", context=ctx) == "u-42"
        assert await reg.resolve("missing", context=ctx) is None
    """

    def __init__(self, 
                 mapping: Mapping[Union[str, Type[BaseModel]], ParameterSource]):
        """Create a new registry from *mapping*.

        Each key can be either a string parameter name **or** a subclass of
        :class:`pydantic.BaseModel`.  For BaseModel keys the corresponding
        parameter name is the *lower-cased* class name (``Address`` →
        ``"address"``).  This makes it possible to group several related
        fields into a single structured object passed to the tool.
        """

        self._mapping: dict[str, ParameterSource] = {}

        for key, src in mapping.items():
            if isinstance(key, str):
                param_name = key
            elif isinstance(key, type) and issubclass(key, BaseModel):
                # Derive parameter name from the model class (Address → address)
                param_name = key.__name__[0].lower() + key.__name__[1:]
            else:
                raise TypeError("ParameterRegistry keys must be str or BaseModel subclasses")

            if param_name in self._mapping:
                raise ValueError(f"Duplicate parameter name '{param_name}' in registry mapping")

            self._mapping[param_name] = src

        self._log = logging.getLogger(name="seqoria_agent.param_registry")

    async def resolve(self, 
                      name: str, 
                      *, 
                      context: ParamContext) -> Optional[Any]:  # noqa: D401
        """Return the resolved value for *name* or *None* when unresolved.

        Resolution order:
        1.  If *name* not found in the registry mapping → *None*.
        2.  If the source is **not callable** → return it as-is.
        3.  If the source **is callable** → inspect its signature and pass the
            subset of *context* kwargs matching the callable parameters.  Both
            sync and async callables are supported.  Exceptions are caught and
            logged, the method returns *None*.
        """
        source = self._mapping.get(name)
        if source is None:
            return None

        if not callable(source):
            return source

        try:
            sig = inspect.signature(obj=source)  # type: ignore[arg-type]
            kwargs = {k: v for k, v in context.items() if k in sig.parameters}
            result = source(**kwargs)  # type: ignore[arg-type]
            if inspect.isawaitable(object=result):  # covers coroutine & awaitable objects
                result = await asyncio.ensure_future(coro_or_future=result)  # type: ignore[assignment]
            return result
        except Exception as exc:  # noqa: BLE001 – user code must not crash core
            self._log.warning("Parameter '%s' failed to resolve: %s", name, exc)
            return None 