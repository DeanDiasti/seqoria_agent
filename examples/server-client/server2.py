"""Example SeqoriaMCP server 2: Math and Time Tools.

This server provides simple utility tools:
- `current_time`: Returns the current server time.
- `divide`: Demonstrates error handling.
- `timestamped_divide`: Tool with dependency on current_time.

To run:
    python examples/server-client/server2.py

Or with uvicorn:
    uvicorn examples.server-client.server2:app --reload --port 8005
"""

from __future__ import annotations

import logging
from datetime import datetime

from seqoria_agent.server import SeqoriaMCP

# ------------------------------------------------------------------
# Application instance
# ------------------------------------------------------------------

# Run on port 8005 to avoid conflict with server1.py (port 8000)
app = SeqoriaMCP("Utility Server", port=8005)
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
    ValueError
        If `b` is zero or if inputs are not valid numbers.
    """
    try:
        a = float(a)
        b = float(b)
    except Exception:
        raise ValueError("a and b must be numbers")

    if b == 0:
        raise ValueError("Parameter 'b' cannot be zero (division by zero)")

    return a / b


# ------------------------------------------------------------------
# Tool with dependency on current_time
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
        raise ValueError("a and b must be numbers")

    result = divide(a, b)
    ts = current_time()
    return f"{ts} → {a} / {b} = {result}"


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(transport="streamable-http")
