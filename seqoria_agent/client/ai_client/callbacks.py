"""Safely invoke optional error callbacks without interrupting tool execution."""

from __future__ import annotations

import logging
from typing import Any, Callable

from seqoria_agent._redaction import redact_text

_logger = logging.getLogger(name="seqoria_agent.callbacks")

__all__: list[str] = ["on_error"]


def _safe_call(cb: Callable[..., None] | None, *args: Any) -> None:  # noqa: D401
    """Invoke *cb* with *args* guarding against non-callable or errors."""
    if not callable(cb):
        return
    try:
        cb(*args)
    except Exception as exc:  # noqa: BLE001 – user callbacks must not crash core
        _logger.warning("Callback raised exception (ignored): %s", redact_text(exc))


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def on_error(error_payload: dict, cb):  # noqa: D401, ANN001
    """Safely forward *error_payload* to *cb* if callable."""
    _safe_call(cb, error_payload)
