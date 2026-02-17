from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from seqoria_agent.memory.config import ToolHandlerConfig


class ToolCallHandler:
    """Track tool outcomes and generate non-destructive guidance."""

    def __init__(self, config: ToolHandlerConfig) -> None:
        self.config = config
        self._attempt_counts: dict[str, int] = defaultdict(int)
        self._recent_guidance: deque[str] = deque(maxlen=20)

    def record_success(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        result: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del result, metadata
        self._attempt_counts[tool_call_id] = 0
        if self.config.store_successful_calls:
            self._recent_guidance.append(f"Tool '{tool_name}' recently succeeded.")

    def record_failure(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        error: str,
        retry_guidance: str | None = None,
    ) -> str | None:
        self._attempt_counts[tool_call_id] += 1
        attempts = self._attempt_counts[tool_call_id]
        guidance = retry_guidance
        if attempts >= self.config.max_retries_before_rewrite and self.config.soft_failure_message:
            soft = (
                f"[Note: Attempted to use {tool_name} but encountered issues. "
                "Proceeding without this step.]"
            )
            self._recent_guidance.append(soft)
            guidance = guidance or soft
        else:
            msg = f"Tool '{tool_name}' failed ({attempts}): {error}"
            self._recent_guidance.append(msg)
            guidance = guidance or msg
        return guidance

    def get_guidance_text(self, max_items: int = 6) -> str | None:
        if not self._recent_guidance:
            return None
        items = list(self._recent_guidance)[-max_items:]
        return "\n".join(f"- {item}" for item in items)

