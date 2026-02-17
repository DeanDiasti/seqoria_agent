from __future__ import annotations

from typing import Any

from seqoria_agent.memory.agent.prompts import (
    MEMORY_COMPRESSED_PROMPT,
    MEMORY_FULL_PROMPT,
    MEMORY_HYBRID_PROMPT,
    MEMORY_SUMMARY_PROMPT,
)


class ResolutionGenerator:
    """Generate compressed/hybrid/full text variants deterministically.

    Each resolution level is wrapped in the corresponding XML-structured
    prompt so that downstream LLM consumers receive consistent formatting
    cues when these records are injected into context.
    """

    def generate(
        self,
        content: str,
        *,
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
        generate_hybrid: bool = True,
        generate_full: bool = True,
    ) -> tuple[str, str | None, str | None]:
        normalized = self._normalize_text(content)
        summary = self._trim_words(self._first_sentence(normalized), 35)
        key_points = self._extract_key_points(normalized, max_points=4)
        metadata_line = self._metadata_line(metadata)
        is_failure = self._detect_failure(normalized)

        # -- Compressed: single-sentence retrieval key ----------------------
        failure_prefix = "Failed: " if is_failure else ""
        compressed = f"[{memory_type}] {failure_prefix}{summary}"
        if metadata_line:
            compressed = f"{compressed} | {metadata_line}"

        # -- Hybrid: medium-detail context record ---------------------------
        hybrid = None
        if generate_hybrid:
            bullets = (
                "\n".join(f"    - {point}" for point in key_points[:3])
                if key_points
                else "    - n/a"
            )
            status_line = self._infer_status_line(normalized, metadata)
            hybrid = (
                f"<summary>{summary}</summary>\n"
                f"<details>\n"
                f"    - type: {memory_type}\n"
                f"{bullets}\n"
                f"    - status: {status_line}\n"
                f"</details>"
            )
            if metadata_line:
                hybrid = f"{hybrid}\n<metadata>{metadata_line}</metadata>"

        # -- Full: complete evidence archive --------------------------------
        full = None
        if generate_full:
            full_bullets = (
                "\n".join(f"    - {point}" for point in key_points)
                if key_points
                else "    - n/a"
            )
            evidence = self._trim_words(normalized, 280)
            risks = self._extract_risks(normalized)
            next_step = self._extract_next_step(normalized)
            full = (
                f"<summary>{summary}</summary>\n"
                f"<structured_details>\n"
                f"    - type: {memory_type}\n"
                f"{full_bullets}\n"
                f"</structured_details>\n"
                f"<evidence>\n"
                f"    \"{evidence}\"\n"
                f"</evidence>\n"
                f"<risks_and_uncertainties>\n"
                f"    - {risks}\n"
                f"</risks_and_uncertainties>\n"
                f"<next_steps>\n"
                f"    - {next_step}\n"
                f"</next_steps>"
            )
            if metadata_line:
                full = f"{full}\n<metadata>{metadata_line}</metadata>"

        return compressed, hybrid, full

    # ------------------------------------------------------------------
    # Text processing helpers
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).strip().split())

    def _trim_words(self, text: str, max_words: int) -> str:
        words = text.strip().split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]) + " ..."

    def _first_sentence(self, text: str) -> str:
        for separator in (". ", "! ", "? "):
            if separator in text:
                return text.split(separator, 1)[0] + separator.strip()
        return text

    def _extract_key_points(self, text: str, *, max_points: int) -> list[str]:
        chunks: list[str] = []
        for separator in (". ", "; ", " - ", "\n"):
            if separator in text:
                chunks = [part.strip(" .-") for part in text.split(separator)]
                break
        if not chunks:
            chunks = [text]
        points = [self._trim_words(chunk, 20) for chunk in chunks if chunk]
        return points[:max_points]

    def _metadata_line(self, metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return ""
        important_keys = ("tool_name", "success", "turn", "source", "scope")
        parts: list[str] = []
        for key in important_keys:
            if key in metadata:
                parts.append(f"{key}={metadata[key]}")
        return ", ".join(parts)

    def _detect_failure(self, text: str) -> bool:
        """Heuristic check for failure signals in normalised text."""
        lower = text.lower()
        failure_signals = ("error", "failed", "failure", "exception", "traceback", "status: failure")
        return any(signal in lower for signal in failure_signals)

    def _infer_status_line(self, text: str, metadata: dict[str, Any] | None) -> str:
        """Derive an explicit status string from content or metadata."""
        if metadata and "success" in metadata:
            return "success" if metadata["success"] else "failure"
        if self._detect_failure(text):
            return "failure"
        lower = text.lower()
        if "success" in lower or "completed" in lower or "resolved" in lower:
            return "success"
        return "recorded"

    def _extract_risks(self, text: str) -> str:
        """Pull risk/uncertainty signals from normalised text."""
        lower = text.lower()
        risk_signals = ("risk", "uncertain", "unknown", "warning", "deprecated", "unstable", "might fail")
        found = [signal for signal in risk_signals if signal in lower]
        if found:
            return f"Signals detected: {', '.join(found)}"
        if self._detect_failure(text):
            return "Failure occurred — root cause may recur."
        return "None identified."

    def _extract_next_step(self, text: str) -> str:
        """Infer follow-up action from normalised text."""
        lower = text.lower()
        next_signals = ("next step", "follow-up", "todo", "then ", "afterwards", "retry")
        for signal in next_signals:
            idx = lower.find(signal)
            if idx != -1:
                snippet = text[idx : idx + 80].strip()
                return self._trim_words(snippet, 20)
        if self._detect_failure(text):
            return "Investigate root cause and retry with corrected approach."
        return "No follow-up required."

