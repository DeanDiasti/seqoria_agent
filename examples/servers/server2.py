"""Example FastXMCP server 2: Math and Time Tools.

This server provides simple utility tools:
- `current_time`: Returns the current server time.
- `divide`: Demonstrates structured `XMCPError` handling.
"""

from __future__ import annotations

from datetime import datetime
import logging

from xmcp.server.fastxmcp import FastXMCP
from xmcp.models.errors import XMCPError

# ------------------------------------------------------------------
# Application instance
# ------------------------------------------------------------------

app = FastXMCP(title="Utility Server")
logger = logging.getLogger(__name__)


@app.tool()
def current_time() -> str:
    """Return the current server time as a formatted string.

    Returns
    -------
    str
        The current time in "YYYY-MM-DD HH:MM:SS" format.
    """

    logger.info("Getting current time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers, `a` by `b`.

    Parameters
    ----------
    a : float
        The numerator.
    b : float
        The denominator. Cannot be zero.

    Returns
    -------
    float
        The result of `a / b`.

    Raises
    ------
    XMCPError
        If `b` is zero, a structured error with code "DIV_BY_ZERO" is raised.
    """

    if b == 0:
        raise XMCPError(
            code="DIV_BY_ZERO",
            detail="Parameter 'b' cannot be zero",
            retryable=False,
            meta={"a": a, "b": b},
        )

    return a / b


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8005) 