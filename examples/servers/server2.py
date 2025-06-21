"""Example FastXMCP server 2: Math and Time Tools.

This server provides simple utility tools:
- `current_time`: Returns the current server time.
- `divide`: Demonstrates structured `XMCPError` handling.
"""

from __future__ import annotations

from datetime import datetime
import logging

from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.models.errors import XMCPError

# ------------------------------------------------------------------
# Application instance
# ------------------------------------------------------------------

app = FastXMCP(title="Utility Server")
logger = logging.getLogger(name=__name__)


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

    try:
        a = float(a)
        b = float(b)
    except Exception:
        raise XMCPError(code="INVALID_ARG", detail="a and b must be numbers", retryable=False)

    if b == 0:
        raise XMCPError(
            code="DIV_BY_ZERO",
            detail="Parameter 'b' cannot be zero",
            retryable=False,
            meta={"a": a, "b": b},
        )

    return a / b


# ------------------------------------------------------------------
# New tool with dependency on current_time -------------------------
# ------------------------------------------------------------------


@app.tool(depends_on=["current_time"])
def timestamped_divide(a: float, b: float) -> str:
    """Divide *a* by *b* and attach the server timestamp.

    Demonstrates a simple dependency so that `current_time` is guaranteed to
    execute before this tool when both are requested together.
    """

    try:
        a = float(a)
        b = float(b)
    except Exception:
        raise XMCPError(code="INVALID_ARG", detail="a and b must be numbers", retryable=False)

    result = divide(a, b)
    ts = current_time()
    return f"{ts} → {a} / {b} = {result}"


# Provide user_id via request header on this server too



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8005) 