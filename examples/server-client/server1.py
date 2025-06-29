"""Example FastXMCP server 1: Basic Tools using the lean FastAPI backend.

Demonstrates:
- Standard @tool (roll_dice)
- Hybrid @display_tool (change_color)
- Display-only tool (flash_screen)
- Structured error handling (flaky_service)
"""

from __future__ import annotations

import random

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.fastxmcp import FastXMCP

# ------------------------------------------------------------------
# Instantiate FastXMCP application
# ------------------------------------------------------------------

app = FastXMCP(title="Agentic Tools Server")

# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

@app.tool()
def roll_dice(n_dice: int) -> list[int]:
    """Roll *n_dice* 6-sided dice and return the results."""
    # Coerce potential string inputs coming from the LLM
    try:
        n_dice = int(n_dice)
    except Exception:
        raise XMCPError(code="INVALID_ARG", detail="n_dice must be an integer", retryable=False)

    if n_dice <= 0:
        raise XMCPError(code="INVALID_ARG", detail="n_dice must be >0", retryable=False)
    if n_dice > 100:
        raise XMCPError(code="LIMIT_EXCEEDED", detail="Cannot roll >100 dice", retryable=False)
    return [random.randint(1, 6) for _ in range(n_dice)]


@app.tool()
def flaky_service() -> str:
    """Randomly fails to demonstrate retryable errors."""
    if random.random() < 0.5:
        raise XMCPError(
            code="SERVICE_UNAVAILABLE",
            detail="Upstream service timed out",
            retryable=True,
            meta={"service": "flaky"},
        )
    return "Flaky service responded successfully"


# ------------------------------------------------------------------
# NEW: Tool that requires a server-populated parameter
# ------------------------------------------------------------------

@app.tool(server_populated=["user_id"])
def personalized_greeting(user_id: str, name: str) -> str:  # noqa: D401
    """Return a greeting that uses the injected ``user_id`` hidden from the LLM."""
    print(f"user_id: {user_id}")
    return f"Hello {name}! (user_id={user_id})"


# ------------------------------------------------------------------
# New tools demonstrating dependency annotations -------------------
# ------------------------------------------------------------------

@app.tool(depends_on=["roll_dice"])
def dice_sum(n_dice: int) -> int:
    """Return the sum of dice rolled by `roll_dice`.

    This simple example is *only* for demonstrating dependency graph execution;
    the function does **not** consume the output of ``roll_dice`` (automatic
    argument injection is pending).  The dependency ensures that the dice are
    rolled *before* this function runs when both are requested in the same
    turn.
    """
    try:
        n_dice = int(n_dice)
    except Exception:
        raise XMCPError(code="INVALID_ARG", detail="n_dice must be an integer", retryable=False)

    return sum(random.randint(1, 6) for _ in range(n_dice))


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8004) 