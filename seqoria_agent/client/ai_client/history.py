"""Default system prompt for application-owned conversation histories."""

from __future__ import annotations


def build_system_prompt(extra_instructions: str | None = None) -> str:  # noqa: D401
    """Return the default system prompt used by *AIClient*.

    The prompt outlines the agent's capabilities and briefly introduces the
    available tools. Callers may supply **extra instructions** that are
    appended *verbatim* as an additional paragraph separated by a newline.
    """

    base_prompt = (
        "You are a helpful and intelligent assistant. You have access to a set of tools "
        "to answer user questions. When you receive a user's query, you must decide whether "
        "to call a tool or to respond directly. If you call a tool, you will receive the result "
        "and must then formulate a final, user-facing response based on that result. "
        "Always provide the final answer in a clear and friendly manner."
    )

    if extra_instructions:
        base_prompt = f"{base_prompt}\n{extra_instructions.strip()}"

    return base_prompt
