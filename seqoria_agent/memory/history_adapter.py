from __future__ import annotations

from typing import Any, Dict, List, Protocol, Union, runtime_checkable

from seqoria_agent.models.chunk import LLMUsage, ToolCallError, ToolCallRequest, ToolCallResult


@runtime_checkable
class MemoryAwareHistory(Protocol):
    """Protocol used by AIClient for memory-aware orchestration."""

    def begin_turn(
        self,
        user_query: str,
        *,
        retrieval_query: str | None = None,
        user_message_content: Union[str, List[Dict[str, Any]], None] = None,
    ) -> list[dict[str, Any]]: ...

    def messages_for_llm(self, query: str | None = None) -> list[dict[str, Any]]: ...

    def record_tool_cycle(
        self,
        tool_calls: list[ToolCallRequest],
        tool_results: list[ToolCallResult | ToolCallError],
    ) -> None: ...

    def commit_assistant_response(
        self,
        content: str,
        *,
        model: str | None = None,
        usage: LLMUsage | None = None,
    ) -> None: ...

    def end_turn(self) -> None: ...

    def append(self, __object: Any) -> None: ...
    def extend(self, __iterable: Any) -> None: ...
    def insert(self, __index: int, __object: Any) -> None: ...


def is_memory_aware_history(history: Any) -> bool:
    """Duck-typed memory-aware check used to avoid strict imports."""
    required = (
        "begin_turn",
        "messages_for_llm",
        "record_tool_cycle",
        "commit_assistant_response",
        "end_turn",
    )
    return all(callable(getattr(history, attr, None)) for attr in required)

