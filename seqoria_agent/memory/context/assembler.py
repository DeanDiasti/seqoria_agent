from __future__ import annotations

from typing import Callable

from seqoria_agent.memory.agent.prompts import (
    CONTEXT_COMPRESSED_HISTORY_HEADER,
    CONTEXT_MEMORY_HEADER,
    CONTEXT_PLAN_HEADER,
    CONTEXT_TOOL_GUIDANCE_HEADER,
)


class ContextAssembler:
    """Assemble final message payload sent to provider APIs.

    Uses XML-structured section headers so the consuming LLM can
    reliably distinguish between base instructions, retrieved memories,
    plan state, tool guidance, and compressed history.
    """

    def assemble(
        self,
        *,
        base_system_prompt: str | None,
        memory_section: str | None,
        plan_section: str | None,
        tool_guidance_section: str | None,
        compressed_history: str | None,
        injected_notes: list[str],
        recent_messages: list[dict],
        total_budget: int,
        token_estimator: Callable[[str], int],
    ) -> list[dict]:
        """Build final ordered context with budget awareness."""
        messages: list[dict] = []
        budget_remaining = total_budget

        # -- Primary system prompt with XML-delimited sections --------------
        system_sections: list[str] = []
        if base_system_prompt:
            system_sections.append(base_system_prompt)
        if memory_section:
            system_sections.append(
                f"{CONTEXT_MEMORY_HEADER}\n{memory_section}"
            )
        if plan_section:
            system_sections.append(
                f"{CONTEXT_PLAN_HEADER}\n{plan_section}"
            )
        if tool_guidance_section:
            system_sections.append(
                f"{CONTEXT_TOOL_GUIDANCE_HEADER}\n{tool_guidance_section}"
            )
        for note in injected_notes:
            system_sections.append(note)

        if system_sections:
            system_text = "\n\n".join(system_sections)
            messages.append({"role": "system", "content": system_text})
            budget_remaining -= token_estimator(system_text)

        # -- Compressed history as a separate system block ------------------
        if compressed_history:
            block = f"{CONTEXT_COMPRESSED_HISTORY_HEADER}\n{compressed_history}"
            cost = token_estimator(block)
            if budget_remaining - cost > 0:
                messages.append({"role": "system", "content": block})
                budget_remaining -= cost

        # -- Recent turns verbatim; clip by remaining budget ----------------
        selected: list[dict] = []
        running = 0
        for message in reversed(recent_messages):
            cost = token_estimator(str(message.get("content", "")))
            if running + cost > max(0, budget_remaining):
                break
            running += cost
            selected.append(message)
        selected.reverse()
        messages.extend(selected)
        return messages

