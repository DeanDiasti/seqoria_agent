from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from seqoria_agent.memory.agent.resolution import ResolutionGenerator
from seqoria_agent.memory.models import MemoryEdge, MemoryNode, Plan


@dataclass(slots=True)
class InferredPlan:
    goal: str
    steps: list[str]
    confidence: float
    rationale: str
    step_metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PlanProgressUpdate:
    step_index: int | None
    status: str
    result_summary: str
    evidence: str
    confidence: float
    blocked_reason: str | None = None
    required_tools: list[str] = field(default_factory=list)


class MemoryAgent:
    """Lightweight memory processing helper.

    The implementation is deterministic and local by default. Callers may swap
    in an external provider later if desired.
    """

    # Root-cause categories used by retry guidance
    _CAUSE_PATTERNS: list[tuple[str, list[str]]] = [
        ("permission_error", ["permission", "forbidden", "403", "unauthorized", "401", "access denied"]),
        ("timeout", ["timeout", "timed out", "deadline exceeded", "context deadline"]),
        ("rate_limit", ["rate limit", "429", "too many requests", "throttl"]),
        ("not_found", ["not found", "404", "no such file", "does not exist", "missing"]),
        ("bad_arguments", ["invalid argument", "invalid param", "type error", "validation", "schema"]),
        ("dependency_error", ["import error", "module not found", "dependency", "package"]),
    ]
    _COMPLETION_MARKERS = (
        "completed",
        "finished",
        "done",
        "implemented",
        "saved",
        "delivered",
        "resolved",
        "executed",
    )
    _BLOCKED_MARKERS = (
        "failed",
        "error",
        "blocked",
        "unable",
        "cannot",
        "can't",
    )

    def __init__(self) -> None:
        self._resolution = ResolutionGenerator()

    async def generate_resolutions(
        self,
        content: str,
        *,
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
        generate_hybrid: bool = True,
        generate_full: bool = True,
    ) -> tuple[str, str | None, str | None]:
        return self._resolution.generate(
            content,
            memory_type=memory_type,
            metadata=metadata,
            generate_hybrid=generate_hybrid,
            generate_full=generate_full,
        )

    async def generate_retry_guidance(
        self,
        *,
        tool_name: str,
        error: str,
        attempt: int,
    ) -> str:
        """Produce structured retry guidance using XML formatting.

        Analyses the error string to classify the root cause and generate
        targeted corrective actions rather than generic advice.
        """
        cause = self._classify_root_cause(error)
        corrective = self._corrective_actions(tool_name, cause, error)

        return (
            f"<retry_guidance>\n"
            f"  <tool>{tool_name}</tool>\n"
            f"  <attempt>{attempt}</attempt>\n"
            f"  <diagnosis>{cause}: {self._trim(error, 120)}</diagnosis>\n"
            f"  <corrective_actions>\n"
            f"{corrective}\n"
            f"  </corrective_actions>\n"
            f"</retry_guidance>"
        )

    async def infer_connection(self, source: MemoryNode, target: MemoryNode) -> MemoryEdge | None:
        """Infer a weak relationship from content overlap."""
        source_terms = set(source.compressed.lower().split())
        target_terms = set(target.compressed.lower().split())
        if not source_terms or not target_terms:
            return None
        overlap = len(source_terms.intersection(target_terms))
        if overlap == 0:
            return None
        weight = min(1.0, 0.2 + overlap / 20.0)
        return MemoryEdge(
            source_id=source.id,
            target_id=target.id,
            relationship="related_to",
            weight=weight,
        )

    def infer_plan_from_turn(
        self,
        *,
        user_query: str,
        existing_plan: Plan | None,
        complexity_min_words: int = 18,
        max_steps: int = 5,
    ) -> InferredPlan | None:
        """Infer a short execution plan when the request looks complex."""
        if existing_plan is not None:
            return None
        normalized = " ".join(user_query.split())
        if not normalized:
            return None
        if not self._looks_complex_request(normalized, complexity_min_words=complexity_min_words):
            return None

        steps = self._extract_steps(normalized, max_steps=max_steps)
        if len(steps) < 2:
            steps = self._fallback_steps(normalized, max_steps=max_steps)

        step_metadata = [self._infer_step_metadata(step) for step in steps]
        confidence = min(0.95, 0.55 + 0.06 * len(steps) + 0.01 * max(0, len(normalized.split()) - complexity_min_words))
        return InferredPlan(
            goal=self._goal_from_query(normalized),
            steps=steps,
            confidence=confidence,
            rationale="Query appears multi-step and benefits from execution scaffolding.",
            step_metadata=step_metadata,
        )

    def infer_step_progress(
        self,
        *,
        plan: Plan,
        tool_calls: list[Any] | None = None,
        tool_results: list[Any] | None = None,
        assistant_text: str | None = None,
    ) -> PlanProgressUpdate | None:
        """Infer plan progression from execution evidence."""
        open_indices = [
            idx
            for idx, step in enumerate(plan.steps)
            if step.status not in {"completed", "skipped"}
        ]
        if not open_indices:
            return None
        current_idx = (
            plan.current_step_index
            if 0 <= plan.current_step_index < len(plan.steps)
            else open_indices[0]
        )

        if tool_calls and tool_results:
            for call, result in zip(tool_calls, tool_results):
                tool_name = self._extract_tool_name(call)
                result_text = self._extract_tool_result_text(result)
                is_error = self._is_error_result(result)
                step_index, explicit_match = self._match_step_for_tool(
                    plan,
                    tool_name=tool_name,
                    fallback_index=current_idx,
                )
                confidence = 0.9 if explicit_match else 0.7
                evidence = f"{tool_name}: {self._trim(result_text, 180)}"
                if is_error:
                    return PlanProgressUpdate(
                        step_index=step_index,
                        status="blocked",
                        result_summary=f"Tool `{tool_name}` failed",
                        evidence=evidence,
                        confidence=confidence,
                        blocked_reason=self._trim(result_text, 160),
                        required_tools=[tool_name],
                    )
                return PlanProgressUpdate(
                    step_index=step_index,
                    status="completed",
                    result_summary=f"Tool `{tool_name}` completed successfully",
                    evidence=evidence,
                    confidence=confidence,
                    required_tools=[tool_name],
                )

        if not assistant_text:
            return None

        normalized = " ".join(assistant_text.split())
        lowered = normalized.lower()
        step_index = self._match_step_for_text(plan, normalized, fallback_index=current_idx)
        if any(marker in lowered for marker in self._BLOCKED_MARKERS):
            return PlanProgressUpdate(
                step_index=step_index,
                status="blocked",
                result_summary="Assistant reported blocker",
                evidence=self._trim(normalized, 180),
                confidence=0.62,
                blocked_reason=self._trim(normalized, 160),
            )
        if any(marker in lowered for marker in self._COMPLETION_MARKERS):
            return PlanProgressUpdate(
                step_index=step_index,
                status="completed",
                result_summary="Assistant reported completion",
                evidence=self._trim(normalized, 180),
                confidence=0.58,
            )
        return None

    def replan_if_blocked(
        self,
        *,
        plan: Plan,
        blocked_step_index: int,
        blocked_reason: str,
        max_steps: int,
    ) -> str | None:
        """Suggest a single recovery step when plan progress is blocked."""
        if len(plan.steps) >= max_steps:
            return None
        reason = self._trim(" ".join(blocked_reason.split()), 90)
        description = f"Resolve blocker from step {blocked_step_index + 1}: {reason}"
        existing = {step.description.lower() for step in plan.steps}
        if description.lower() in existing:
            return None
        return description

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _looks_complex_request(self, text: str, *, complexity_min_words: int) -> bool:
        lowered = text.lower()
        words = len(text.split())
        if words >= complexity_min_words:
            return True
        if "\n" in text and any(marker in text for marker in ("- ", "* ", "1.", "1)")):
            return True
        multi_step_markers = (
            " then ",
            " after ",
            " before ",
            " compare ",
            " and ",
            " while ",
            " step",
            " workflow",
            " plan",
        )
        marker_hits = sum(1 for marker in multi_step_markers if marker in lowered)
        return marker_hits >= 2

    def _extract_steps(self, query: str, *, max_steps: int) -> list[str]:
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        bullet_steps = []
        for line in lines:
            if re.match(r"^([-*]|\d+[.)])\s+", line):
                bullet_steps.append(re.sub(r"^([-*]|\d+[.)])\s+", "", line).strip())
        if bullet_steps:
            return bullet_steps[:max_steps]

        inline_numbered = re.findall(r"\d+\)\s*(.*?)(?=(?:\s+\d+\)|$))", query)
        cleaned_inline = [
            self._normalize_step_text(part)
            for part in inline_numbered
            if self._normalize_step_text(part)
        ]
        if len(cleaned_inline) >= 2:
            return cleaned_inline[:max_steps]

        chunks = re.split(r"\b(?:and then|then|after that|next|finally)\b|[;\n]", query, flags=re.IGNORECASE)
        cleaned = [self._normalize_step_text(chunk) for chunk in chunks if self._normalize_step_text(chunk)]
        if len(cleaned) >= 2:
            return cleaned[:max_steps]

        if " and " in query.lower():
            and_chunks = [self._normalize_step_text(part) for part in re.split(r"\band\b", query, flags=re.IGNORECASE)]
            and_chunks = [part for part in and_chunks if part]
            if len(and_chunks) >= 2:
                return and_chunks[:max_steps]
        return cleaned[:max_steps]

    def _fallback_steps(self, query: str, *, max_steps: int) -> list[str]:
        fallback = [
            f"Clarify objective and constraints for: {self._trim(query, 80)}",
            "Execute required actions and gather evidence",
            "Summarize outcomes and provide next actions",
        ]
        return fallback[:max_steps]

    def _goal_from_query(self, query: str) -> str:
        return self._trim(query.rstrip(". "), 180)

    def _normalize_step_text(self, text: str) -> str:
        cleaned = " ".join(text.split()).strip(" -:;,.")
        if not cleaned:
            return ""
        if len(cleaned) < 8:
            return ""
        return cleaned[0].upper() + cleaned[1:]

    def _infer_step_metadata(self, step_text: str) -> dict[str, Any]:
        explicit_tools: list[str] = []
        for match in re.findall(r"(?:use|call|run)\s+`?([a-zA-Z_][\w-]*)`?\s+tool", step_text, flags=re.IGNORECASE):
            explicit_tools.append(match.strip())
        fn_matches = re.findall(r"([a-zA-Z_][\w-]*)\s*\(", step_text)
        for fn_name in fn_matches:
            explicit_tools.append(fn_name.strip())
        required_tools = list(dict.fromkeys(tool for tool in explicit_tools if tool))
        return {"required_tools": required_tools, "metadata": {"source": "auto_infer"}}

    def _extract_tool_name(self, tool_call: Any) -> str:
        for attr in ("name", "function_name"):
            if hasattr(tool_call, attr):
                value = getattr(tool_call, attr)
                if isinstance(value, str) and value:
                    return value
        if isinstance(tool_call, dict):
            for key in ("name", "function_name"):
                value = tool_call.get(key)
                if isinstance(value, str) and value:
                    return value
        return "unknown_tool"

    def _extract_tool_result_text(self, tool_result: Any) -> str:
        if hasattr(tool_result, "error"):
            value = getattr(tool_result, "error")
            if value is not None:
                return str(value)
        if hasattr(tool_result, "result"):
            value = getattr(tool_result, "result")
            if value is not None:
                return str(value)
        if isinstance(tool_result, dict):
            if "error" in tool_result:
                return str(tool_result["error"])
            if "result" in tool_result:
                return str(tool_result["result"])
        return str(tool_result)

    def _is_error_result(self, tool_result: Any) -> bool:
        role = getattr(tool_result, "role", None)
        if role == "tool_error":
            return True
        if hasattr(tool_result, "error") and getattr(tool_result, "error"):
            return True
        if isinstance(tool_result, dict):
            if tool_result.get("role") == "tool_error":
                return True
            if bool(tool_result.get("error")):
                return True
        return False

    def _match_step_for_tool(
        self,
        plan: Plan,
        *,
        tool_name: str,
        fallback_index: int,
    ) -> tuple[int, bool]:
        lowered_tool = tool_name.lower()
        for idx, step in enumerate(plan.steps):
            if lowered_tool in {name.lower() for name in step.required_tools}:
                return idx, True
        for idx, step in enumerate(plan.steps):
            if lowered_tool in step.description.lower():
                return idx, True
        return fallback_index, False

    def _match_step_for_text(self, plan: Plan, text: str, *, fallback_index: int) -> int:
        text_terms = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
        best_idx = fallback_index
        best_score = 0
        for idx, step in enumerate(plan.steps):
            if step.status in {"completed", "skipped"}:
                continue
            step_terms = set(re.findall(r"[a-zA-Z0-9_]+", step.description.lower()))
            score = len(text_terms.intersection(step_terms))
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _classify_root_cause(self, error: str) -> str:
        lower = error.lower()
        for cause, patterns in self._CAUSE_PATTERNS:
            if any(pattern in lower for pattern in patterns):
                return cause
        return "unknown"

    def _corrective_actions(self, tool_name: str, cause: str, error: str) -> str:
        """Return numbered corrective actions based on root cause category."""
        actions: list[str] = []
        if cause == "permission_error":
            actions = [
                "Verify credentials and access permissions for the target resource.",
                "Try authenticating with a different scope or token.",
            ]
        elif cause == "timeout":
            actions = [
                "Retry with a longer timeout or smaller payload.",
                "Check if the target service is healthy before retrying.",
            ]
        elif cause == "rate_limit":
            actions = [
                "Wait before retrying — apply exponential backoff.",
                "Reduce request frequency or batch operations.",
            ]
        elif cause == "not_found":
            actions = [
                "Verify the resource path, name, or identifier is correct.",
                "Check if the resource was recently moved or deleted.",
            ]
        elif cause == "bad_arguments":
            actions = [
                "Review and correct the argument types and values.",
                "Consult the tool schema for required parameters.",
            ]
        elif cause == "dependency_error":
            actions = [
                "Ensure the required dependency is installed and importable.",
                "Check version compatibility with the current environment.",
            ]
        else:
            actions = [
                f"Inspect the full error output from '{tool_name}' for clues.",
                "Try an alternative tool or approach if available.",
            ]
        numbered = "\n".join(f"    {i}. {action}" for i, action in enumerate(actions, 1))
        return numbered

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

