"""Conversation history utilities for *seqoria_agent*'s AIClient stack.

This module centralises helpers that operate on the *history* list shared
between the high-level :class:`seqoria_agent.client.AIClient`,
:class:`seqoria_agent.client.ai_client.Executor` and their collaborators.

Currently the helpers are intentionally *simple* – sophistication can be added
in later phases without touching the surrounding code.
"""

from __future__ import annotations

import logging
from typing import List

# Prefer structlog where available to keep logging style consistent across the
# codebase.  Fallback to stdlib `logging` otherwise so we never break.
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    structlog = None

_logger = (
    structlog.get_logger("seqoria_agent.history")  # type: ignore[assignment]
    if structlog is not None
    else logging.getLogger("seqoria_agent.history")
)

# ---------------------------------------------------------------------------
# System-prompt helper
# ---------------------------------------------------------------------------

def build_system_prompt(extra_instructions: str | None = None) -> str:  # noqa: D401
    """Return the default system prompt used by *AIClient*.

    The prompt outlines the agent's capabilities and briefly introduces the
    available meta-tools.  Callers may supply **extra instructions** that are
    appended *verbatim* as an additional paragraph separated by a newline.
    """

    base_prompt = (
        "You are a helpful and intelligent assistant. You have access to a set of tools "
        "to answer user questions. When you receive a user's query, you must decide whether "
        "to call a tool or to respond directly. If you call a tool, you will receive the result "
        "and must then formulate a final, user-facing response based on that result. "
        "Always provide the final answer in a clear and friendly manner."
        "\nIf you need to reason through intermediate steps before the next tool, call the `__think__` tool with your thought."
        "\nIf the conversation has grown long, you may call the `__summarize__` tool to compress the history before continuing."
        "\nIf you need additional information from the user, call the `__clarify__` tool with a concise question."
    )

    if extra_instructions:
        base_prompt = f"{base_prompt}\n{extra_instructions.strip()}"

    return base_prompt


# ---------------------------------------------------------------------------
# History summarisation helper
# ---------------------------------------------------------------------------

def summarise_history(history: List[dict]) -> str:  # noqa: D401
    """Return a naive summary of *history* by concatenating the last few messages.

    The implementation is intentionally simplistic – it merely prints the last
    ``N`` conversation turns so that Phase 5 does not introduce behavioural
    changes.  A more sophisticated summariser can replace this later without
    affecting callers.
    """

    if not history:
        return "(no history)"

    N = 6  # Keep this in sync with previous hard-coded value used in Phase 4.
    recent_msgs = history[-N:]
    parts: list[str] = []
    for msg in recent_msgs:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")

    summary = "\n".join(parts)

    _logger.debug("History summarised", original_len=len(history), returned_len=len(summary))

    return summary 