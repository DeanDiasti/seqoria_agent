from __future__ import annotations

"""Parameter resolution utilities for *Server-Populated Parameters* (SPP)."""

from typing import Any, Awaitable, Callable, Mapping, Optional, Union, Type
import inspect
import logging
import asyncio
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
        parameter name uses a lower-case first letter (``AddressInfo`` →
        ``"addressInfo"``).  This makes it possible to group several related
        fields into a single structured object passed to the tool.
        """

        self._mapping: dict[str, ParameterSource] = {}
        # Precomputed callable parameter filters for fast resolution.
        # - None: callable accepts **kwargs → pass full context
        # - set(): callable takes no kwargs (or signature unknown) → call without kwargs
        # - set({...}): pass only these keys (intersection with context)
        self._kw_params: dict[str, set[str] | None] = {}

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

            # Precompute signature-derived kw filtering once (avoid per-call inspect.signature).
            if callable(src):
                try:
                    sig = inspect.signature(obj=src)  # type: ignore[arg-type]
                    params = sig.parameters.values()
                    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                        self._kw_params[param_name] = None
                    else:
                        self._kw_params[param_name] = {
                            p.name
                            for p in params
                            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                        }
                except Exception:
                    # If we can't introspect the callable, fall back to calling it without kwargs.
                    self._kw_params[param_name] = set()

            self._mapping[param_name] = src

        self._log = logging.getLogger(name="seqoria_agent.param_registry")

    async def resolve(self,
                      name: str,
                      *,
                      context: ParamContext) -> Optional[Any]:  # noqa: D401
        """Return the resolved value for *name* or *None* when unresolved.

        Resolution order:
        1.  A value in trusted request *context* takes precedence.
        2.  If *name* is absent from the registry mapping → *None*.
        3.  If the source is **not callable** → return it as-is.
        4.  If the source **is callable** → inspect its signature and pass the
            subset of *context* kwargs matching the callable parameters.  Both
            sync and async callables are supported.  Exceptions are caught and
            logged, the method returns *None*.
        """
        # ------------------------------------------------------------------
        # 1. Per-request literal override ------------------------------------
        # ------------------------------------------------------------------
        if name in context:
            return context[name]

        # 2. Registry lookup --------------------------------------------------
        source = self._mapping.get(name)
        if source is None:
            return None

        try:
            if not callable(source):
                return source

            kw_params = self._kw_params.get(name, set())
            if kw_params is None:
                # Callable accepts **kwargs: pass full context mapping.
                kwargs = dict(context)
            elif not kw_params:
                # No kwargs expected (or signature unknown): call without kwargs.
                kwargs = {}
            else:
                # Pass only the subset of context keys the callable declares.
                kwargs = {k: context[k] for k in kw_params if k in context}

            if inspect.iscoroutinefunction(source):
                result = source(**kwargs)
            else:
                result = await asyncio.to_thread(source, **kwargs)

            if inspect.isawaitable(object=result):  # covers coroutine & awaitable objects
                return await result  # type: ignore[misc]
            return result
        except Exception as exc:  # noqa: BLE001 – user code must not crash core
            self._log.warning("Parameter '%s' failed to resolve: %s", name, exc)
            return None
