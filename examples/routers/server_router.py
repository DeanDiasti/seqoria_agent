"""Example SeqoriaMCP server demonstrating tool grouping.

Run with:
    python examples/routers/server_router.py

This server defines math-related tools with dependency annotations.
"""

from __future__ import annotations

from seqoria_agent.server import SeqoriaMCP

# ------------------------------------------------------------------
# Create SeqoriaMCP server
# ------------------------------------------------------------------

app = SeqoriaMCP("Math Tools Server")


# ------------------------------------------------------------------
# Define math tools
# ------------------------------------------------------------------


@app.tool()
def add(a: int, b: int) -> int:
    """Return a + b."""
    try:
        return int(a) + int(b)
    except Exception as exc:
        raise ValueError(f"Invalid arguments: {exc}")


@app.tool()
def multiply(a: int, b: int) -> int:
    """Return a * b."""
    try:
        return int(a) * int(b)
    except Exception as exc:
        raise ValueError(f"Invalid arguments: {exc}")


@app.tool(depends_on=["add", "multiply"])
def add_then_multiply(a: int, b: int, c: int) -> int:
    """Compute (a + b) x c.

    This tool depends on add and multiply, so if the LLM requests all three
    in a single turn, add and multiply will run first (in parallel), then
    this tool will run.
    """
    return (int(a) + int(b)) * int(c)


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(transport="streamable-http")
