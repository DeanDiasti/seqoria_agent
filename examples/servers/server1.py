"""Example FastXMCP server 1: Basic Tools using the lean FastAPI backend.

Demonstrates:
- Standard @tool (roll_dice)
- Hybrid @display_tool (change_color)
- Display-only tool (flash_screen)
- Structured error handling (flaky_service)
"""

from __future__ import annotations

import random

from xmcp.models.envelope import DisplayEnvelope
from xmcp.models.errors import XMCPError
from xmcp.server.fastxmcp import FastXMCP

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

    if n_dice <= 0:
        raise XMCPError(code="INVALID_ARG", detail="n_dice must be >0", retryable=False)
    if n_dice > 100:
        raise XMCPError(code="LIMIT_EXCEEDED", detail="Cannot roll >100 dice", retryable=False)
    return [random.randint(1, 6) for _ in range(n_dice)]


_COLOR_HEX = {
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "pink": "#FFC0CB",
    "purple": "#800080",
    "orange": "#FFA500",
}


@app.display_tool()
def change_color(color: str, show_user: bool = False):
    """Hybrid tool that either returns a HEX code or sends a UI notification."""

    hex_code = _COLOR_HEX.get(color.lower())
    if hex_code is None:
        raise XMCPError(code="UNKNOWN_COLOR", detail=f"Unknown colour '{color}'", retryable=False)

    if show_user:
        return DisplayEnvelope(
            type="notification",
            title="Colour changed",
            payload=f"Background colour set to **{color}** ({hex_code})",
        )

    return hex_code


@app.display_tool()
def flash_screen(color: str, times: int = 3, show_user: bool = False):
    """Display-only tool that flashes the screen."""

    return DisplayEnvelope(
        type="notification",
        title="Screen flash",
        payload=f"Flashing screen {times} × with {color}",
        meta={"times": times, "color": color},
    )


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
# Entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8004) 