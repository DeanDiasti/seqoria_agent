from __future__ import annotations

"""Example FastXMCP server demonstrating *router* usage.

Run with:
    uvicorn examples.servers.server_router:app --reload --port 8006

This server mounts a ``FastXMCPRouter`` which groups math-related tools under
``/math`` HTTP prefix (irrelevant for MCP usage) while exposing the tools to
LLMs via the normal MCP interface.
"""

from seqoria_agent.server.fastxmcp import FastXMCP, FastXMCPRouter
from seqoria_agent.models.errors import XMCPError

# ------------------------------------------------------------------
# Create router and define tools
# ------------------------------------------------------------------

math_router = FastXMCPRouter(prefix="/math", tags=["math"])


@math_router.tool()
def add(a: int, b: int) -> int:
    """Return *a + b*."""
    try:
        return int(a) + int(b)
    except Exception as exc:  # pragma: no cover – input validation for demo
        raise XMCPError(code="INVALID_ARG", detail=str(exc), retryable=False)


@math_router.tool()
def multiply(a: int, b: int) -> int:
    """Return *a * b*."""
    try:
        return int(a) * int(b)
    except Exception as exc:
        raise XMCPError(code="INVALID_ARG", detail=str(exc), retryable=False)


@math_router.tool(depends_on=["add", "multiply"])
def add_then_multiply(a: int, b: int, c: int) -> int:
    """Compute *(a + b) × c* to showcase dependency annotations (for LLM).

    The function executes after ``add`` and ``multiply`` if the LLM asks for all
    three in a single turn, letting the client parallelise the first two.
    """
    return (int(a) + int(b)) * int(c)


# ------------------------------------------------------------------
# Build main FastXMCP app and mount router
# ------------------------------------------------------------------

app = FastXMCP(title="Math Router Server")
app.include_router(math_router)


# ------------------------------------------------------------------
# Entry-point helper for local runs
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8006) 