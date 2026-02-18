from __future__ import annotations

from typing import Any

from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatMessages,
    SystemMessage,
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
    UserMessage,
)

from seqoria_agent.memory.config import ContextConfig


def estimate_tokens(text: str) -> int:
    """Very lightweight token estimate suitable for budget decisions."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_to_dict(msg: Any) -> dict[str, Any]:
    """Normalize supported message models to dict form."""
    if isinstance(msg, dict):
        return msg
    if isinstance(msg, (SystemMessage, AssistantMessage, UserMessage)):
        return {"role": msg.role, "content": msg.content}
    if isinstance(msg, ToolCallRequest):
        return {
            "role": "tool_call",
            "id": msg.id,
            "type": "function_call",
            "name": msg.name,
            "arguments": msg.arguments or {},
        }
    if isinstance(msg, ToolCallResult):
        return {
            "role": "tool_result",
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
            "result": msg.result,
            "type": msg.type,
        }
    if isinstance(msg, ToolCallError):
        return {
            "role": "tool_error",
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
            "error": msg.error,
            "result": msg.error,
            "type": msg.type,
            "is_error": True,
        }
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "__dict__"):
        return dict(msg.__dict__)
    return {"role": "system", "content": str(msg)}


class ContextManager:
    """Canonical timeline + short-term context manager."""

    def __init__(self, config: ContextConfig) -> None:
        self.config = config
        self.raw_messages: list[dict[str, Any]] = []
        self.compressed_history: str | None = None
        self._base_system_prompt: str | None = None
        self._injected_notes: list[str] = []

    @property
    def base_system_prompt(self) -> str | None:
        return self._base_system_prompt

    def set_base_system_prompt(self, content: str | None) -> None:
        """Set or clear the base system prompt used for context assembly."""
        if content is None:
            self._base_system_prompt = None
            return
        normalized = content.strip()
        self._base_system_prompt = normalized or None

    @property
    def total_tokens(self) -> int:
        return sum(estimate_tokens(str(message.get("content", ""))) for message in self.raw_messages)

    def add_message(self, message: dict[str, Any] | ChatMessages) -> None:
        data = message_to_dict(message)
        role = data.get("role")
        if role == "system" and isinstance(data.get("content"), str):
            self._base_system_prompt = data["content"]
        self.raw_messages.append(data)

    def insert_message(self, index: int, message: Any) -> None:
        self.raw_messages.insert(index, message_to_dict(message))

    def set_message(self, index: int, message: Any) -> None:
        self.raw_messages[index] = message_to_dict(message)

    def delete_message(self, index: int) -> None:
        del self.raw_messages[index]

    def latest_user_text(self) -> str | None:
        for message in reversed(self.raw_messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    def get_recent_messages(self, max_tokens: int | None = None) -> list[dict[str, Any]]:
        """Return most recent messages within configured limits."""
        max_messages = self.config.max_recent_messages
        # System content is assembled separately; keeping it here duplicates prompt blocks.
        candidates = [message for message in self.raw_messages if message.get("role") != "system"]
        recent = candidates[-max_messages:] if max_messages > 0 else list(candidates)
        if max_tokens is None:
            max_tokens = self.config.max_recent_tokens

        selected: list[dict[str, Any]] = []
        running = 0
        for message in reversed(recent):
            token_cost = estimate_tokens(str(message.get("content", "")))
            if running + token_cost > max_tokens:
                break
            running += token_cost
            selected.append(message)
        selected.reverse()
        return selected

    def older_messages_for_compression(self) -> list[dict[str, Any]]:
        """Return message window eligible for compression."""
        if len(self.raw_messages) <= self.config.max_recent_messages:
            return []
        cut_index = len(self.raw_messages) - self.config.max_recent_messages
        return self.raw_messages[:cut_index]

    def set_compressed_history(self, summary: str | None) -> None:
        self.compressed_history = summary

    def inject_system_note(self, note: str) -> None:
        self._injected_notes.append(note)

    def consume_injected_notes(self) -> list[str]:
        notes = list(self._injected_notes)
        self._injected_notes.clear()
        return notes

