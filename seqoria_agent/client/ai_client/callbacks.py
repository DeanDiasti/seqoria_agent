"""Safe wrapper helpers for optional callbacks used by *AIClient*.

External users of :class:`seqoria_agent.client.AIClient` can supply optional
synchronous callbacks like ``on_display`` to intercept display envelopes or
``on_error`` to observe structured error payloads.  Because these callbacks are
**user code** they may raise exceptions – the helpers defined here encapsulate
common safety patterns so that failures never propagate back into the core
execution engine.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    structlog = None

_logger = (
    structlog.get_logger("seqoria_agent.callbacks")  # type: ignore[assignment]
    if structlog is not None
    else logging.getLogger("seqoria_agent.callbacks")
)

__all__: list[str] = [
    "on_display",
    "on_error",
    "on_metadata",
]


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _safe_call(cb: Optional[Callable[[Any, List[dict] | None], None]], *args):  # noqa: D401
    """Invoke *cb* with *args* guarding against non-callable or errors."""
    if not callable(cb):
        return
    try:
        cb(*args)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 – user callbacks must not crash core
        _logger.warning("Callback raised exception", error=str(exc), callback=getattr(cb, "__name__", str(cb)))


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def on_display(envelope: dict, history: list[dict] | None, cb):  # noqa: D401, ANN001
    """Safely forward *envelope* to *cb* if callable."""
    _safe_call(cb, envelope, history)


def on_error(error_payload: dict, history: list[dict] | None, cb):  # noqa: D401, ANN001
    """Safely forward *error_payload* to *cb* if callable."""
    _safe_call(cb, error_payload, history)


def on_metadata(meta: dict, history: list[dict] | None, cb):  # noqa: D401, ANN001
    """Safely forward *meta* block to *cb* if callable."""
    _safe_call(cb, meta, history) 