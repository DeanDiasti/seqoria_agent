"""Example SeqoriaMCP server: Basic Tools using MCP SDK.

Demonstrates:
- Standard @tool (roll_dice)
- Server-populated parameters (personalized_greeting)
- Tool dependencies (dice_sum depends on roll_dice)
- Error handling

To run:
    python examples/server-client/server1.py

Or with uvicorn:
    uvicorn examples.server-client.server1:app --reload --port 8004
"""

from __future__ import annotations

import random

from seqoria_agent.server import SeqoriaMCP

# ------------------------------------------------------------------
# Instantiate SeqoriaMCP application
# ------------------------------------------------------------------

app = SeqoriaMCP("Agentic Tools Server")


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------


@app.tool()
def roll_dice(n_dice: int) -> list[int]:
    """Roll n_dice 6-sided dice and return the results."""
    try:
        n_dice = int(n_dice)
    except Exception:
        raise ValueError("n_dice must be an integer")

    if n_dice <= 0:
        raise ValueError("n_dice must be > 0")
    if n_dice > 100:
        raise ValueError("Cannot roll > 100 dice")

    return [random.randint(1, 6) for _ in range(n_dice)]


@app.tool()
def flaky_service() -> str:
    """Randomly fails to demonstrate retryable errors."""
    if random.random() < 0.5:
        raise RuntimeError("Upstream service timed out")
    return "Flaky service responded successfully"


# ------------------------------------------------------------------
# Tool with server-populated parameter
# ------------------------------------------------------------------


@app.tool(server_populated=["user_id"])
def personalized_greeting(name: str, user_id: str) -> str:
    """Return a greeting using the injected user_id hidden from the LLM."""
    return f"Hello {name}! (user_id={user_id})"


# Register param provider for user_id
@app.register_param_provider
def provide_user_id(ctx, tool_name: str, param_name: str):
    """Provide user_id from request context or use default."""
    if param_name == "user_id":
        return "demo-user-123"
    return None


# ------------------------------------------------------------------
# Tool with dependency annotation
# ------------------------------------------------------------------


@app.tool(depends_on=["roll_dice"])
def dice_sum(n_dice: int) -> int:
    """Return the sum of dice rolled.

    This demonstrates dependency graph execution - dice are rolled
    before this function runs when both are requested together.
    """
    try:
        n_dice = int(n_dice)
    except Exception:
        raise ValueError("n_dice must be an integer")

    return sum(random.randint(1, 6) for _ in range(n_dice))


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Run using MCP SDK's built-in server
    app.run(transport="streamable-http")
