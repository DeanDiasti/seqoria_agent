"""Safe wrapper helpers for optional callbacks used by *AIClient*.

External users of :class:`seqoria_agent.client.AIClient` can supply optional
synchronous callbacks like ``on_metadata`` to inspect tool metadata or
``on_error`` to observe structured error payloads. Because these callbacks are
**user code** they may raise exceptions – the helpers defined here encapsulate
common safety patterns so that failures never propagate back into the core
execution engine.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

_logger = logging.getLogger(name="seqoria_agent.callbacks")

__all__: list[str] = ["on_error", "on_metadata"]


def _safe_call(cb: Optional[Callable[[Any, List[dict] | None], None]], *args) -> None:  # noqa: D401
    """Invoke *cb* with *args* guarding against non-callable or errors."""
    if not callable(cb):
        return
    try:
        cb(*args)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 – user callbacks must not crash core
        _logger.warning("Callback raised exception (ignored): %s", exc)


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def on_error(error_payload: dict, cb):  # noqa: D401, ANN001
    """Safely forward *error_payload* to *cb* if callable."""
    _safe_call(cb, error_payload)


def on_metadata(meta: dict, cb):  # noqa: D401, ANN001
    """Safely forward *meta* block to *cb* if callable."""
    _safe_call(cb, meta)
