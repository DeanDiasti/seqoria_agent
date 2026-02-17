from __future__ import annotations

from typing import Any


class Compressor:
    """Deterministic text compressor for earlier conversation windows."""

    def __init__(self, strategy: str = "hierarchical") -> None:
        self.strategy = strategy

    def compress(
        self,
        messages: list[dict[str, Any]],
        *,
        existing_summary: str | None = None,
        max_lines: int = 12,
    ) -> str:
        """Create a compact summary from older messages."""
        if not messages and existing_summary:
            return existing_summary
        if not messages:
            return ""

        if self.strategy == "hierarchical":
            return self._hierarchical_compress(
                messages,
                existing_summary=existing_summary,
                max_lines=max_lines,
            )

        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = self._normalize_content(message.get("content", ""))
            if not content:
                continue
            short = self._truncate_words(content, 24)
            lines.append(f"- {role}: {short}")
            if len(lines) >= max_lines:
                break

        summary = "\n".join(lines)
        if not existing_summary:
            return summary

        if self.strategy == "progressive":
            # Keep previous summary and append latest chunk summary.
            merged = [existing_summary.strip(), summary.strip()]
            return "\n".join(part for part in merged if part)

        if self.strategy == "selective":
            # Prefer newest summary for selective strategy.
            return summary

        # summarize_chunks fallback combines both sections.
        return "\n".join(part for part in [existing_summary.strip(), summary.strip()] if part)

    def _hierarchical_compress(
        self,
        messages: list[dict[str, Any]],
        *,
        existing_summary: str | None,
        max_lines: int,
    ) -> str:
        """Two-stage summary: interaction summaries, then usage-level overview."""
        interaction_summaries: list[str] = []
        role_counts: dict[str, int] = {}
        intent_hints: list[str] = []
        outcome_hints: list[str] = []

        for message in messages:
            role = str(message.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            content = self._normalize_content(message.get("content", ""))
            if not content:
                continue

            short = self._truncate_words(content, 20)
            interaction_summaries.append(f"- {role}: {short}")
            if len(interaction_summaries) >= max_lines:
                break

            lower = content.lower()
            if role == "user" and len(intent_hints) < 4:
                intent_hints.append(self._truncate_words(content, 12))
            if role in {"assistant", "tool_result", "tool_error"} and len(outcome_hints) < 4:
                if "error" in lower or "failed" in lower:
                    outcome_hints.append(f"failure: {self._truncate_words(content, 10)}")
                else:
                    outcome_hints.append(self._truncate_words(content, 10))

        role_summary = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        intent_summary = "; ".join(intent_hints[:3]) if intent_hints else "n/a"
        outcome_summary = "; ".join(outcome_hints[:3]) if outcome_hints else "n/a"

        sections: list[str] = []
        if existing_summary:
            sections.append(
                "[PREVIOUS USAGE SUMMARY]\n"
                f"- carried_forward: {self._truncate_words(existing_summary.strip(), 36)}"
            )

        sections.append(
            "[INTERACTION SUMMARIES]\n"
            + ("\n".join(interaction_summaries) if interaction_summaries else "- none")
        )
        sections.append(
            "[USAGE SUMMARY]\n"
            f"- role_mix: {role_summary or 'n/a'}\n"
            f"- user_intents: {intent_summary}\n"
            f"- outcomes: {outcome_summary}"
        )
        return "\n\n".join(sections)

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_chunks.append(str(item.get("text", "")))
                    else:
                        text_chunks.append(str(item))
                else:
                    text_chunks.append(str(item))
            return " ".join(text_chunks).strip()
        return str(content).strip()

    def _truncate_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ..."

