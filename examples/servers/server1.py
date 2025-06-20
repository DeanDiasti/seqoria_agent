"""Example XMCP server 1: Basic Tools.

This server demonstrates:
- A standard `@mcp.tool` (`roll_dice`) that returns data for the LLM.
- A `@mcp.display_tool` (`change_color`) with hybrid LLM/UI return paths.
- A display-only tool (`flash_screen`).
- A tool that raises a structured, retryable `XMCPError` (`flaky_service`).
"""

import random
import time
from mcpx import DisplayEnvelope
from mcpx.server.runtime import XMCP
from mcpx.models.errors import XMCPError

# Example XMCP server demonstrating standard and display tools.

mcp = XMCP(name="Agentic Tools Server", strict_mode=True)


@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results.

    Parameters
    ----------
    n_dice : int
        The number of dice to roll. Must be between 1 and 100.

    Returns
    -------
    list[int]
        A list of integers representing the result of each die roll.
    
    Raises
    ------
    XMCPError
        If `n_dice` is not a positive integer or exceeds 100.
    """
    if n_dice <= 0:
        raise XMCPError(code="INVALID_ARG", detail="n_dice must be a positive integer", retryable=False)
    if n_dice > 100:
        raise XMCPError(code="LIMIT_EXCEEDED", detail="Cannot roll more than 100 dice", retryable=False)
    return [random.randint(1, 6) for _ in range(n_dice)]


# ------------------------------------------------------------------
# Colour helper mapping (basic)
# ------------------------------------------------------------------

_COLOR_HEX: dict[str, str] = {
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


@mcp.display_tool
def change_color(color: str, show_user: bool = False):
    """Change the (imaginary) UI background colour or return its HEX code.

    If ``show_user`` is ``True`` the caller/client intends this as a *display*
    action and we therefore return a :class:`DisplayEnvelope` so the
    front-end can react instantly. Otherwise we simply return the hex code so
    the LLM can embed it in its textual answer.
    """

    hex_code = _COLOR_HEX.get(color.lower())
    if hex_code is None:
        raise XMCPError(code="UNKNOWN_COLOR", detail=f"Colour '{color}' not recognised", retryable=False)

    if show_user:
        return DisplayEnvelope(
            type="notification",
            title="Colour changed",
            payload=f"Background colour set to **{color}** ({hex_code})",
        )

    return hex_code


@mcp.display_tool
def flash_screen(color: str, times: int = 3, show_user: bool = False):
    """Flash the screen *times* with *color* (display-only)."""

    return DisplayEnvelope(
        type="notification",
        title="Screen flash",
        payload=f"Flashing screen {times} × with {color}",
        meta={"times": times, "color": color},
    )


@mcp.tool
def flaky_service() -> str:
    """Simulate a transient service failure to demonstrate `retryable` errors."""

    if random.random() < 0.5:
        raise XMCPError(
            code="SERVICE_UNAVAILABLE",
            detail="Upstream service timed out",
            retryable=True,
            meta={"service": "flaky"},
        )

    return "Flaky service responded successfully"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005, path="/mcp") 