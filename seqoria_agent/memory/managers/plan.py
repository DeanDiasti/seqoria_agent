from __future__ import annotations

import uuid
from typing import Any, Literal

from seqoria_agent.memory.models import Plan, PlanStep


class PlanManager:
    """Simple in-memory plan manager."""

    def __init__(self) -> None:
        self._current_plan: Plan | None = None

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    def set_plan(
        self,
        goal: str,
        steps: list[str],
        *,
        source: Literal["inferred", "explicit"] = "explicit",
        step_metadata: list[dict[str, Any]] | None = None,
    ) -> Plan:
        normalized_steps = [description.strip() for description in steps if description.strip()]
        if not normalized_steps:
            normalized_steps = ["Clarify objective and constraints", "Execute tasks", "Synthesize outcome"]

        plan_steps: list[PlanStep] = []
        for idx, description in enumerate(normalized_steps):
            raw_meta = (step_metadata[idx] if step_metadata and idx < len(step_metadata) else {}) or {}
            required_tools_raw = raw_meta.get("required_tools") or []
            required_tools = [str(item).strip() for item in required_tools_raw if str(item).strip()]
            extra_meta = raw_meta.get("metadata") if isinstance(raw_meta.get("metadata"), dict) else {}
            plan_steps.append(
                PlanStep(
                    id=str(uuid.uuid4()),
                    description=description,
                    status="pending",
                    required_tools=required_tools,
                    metadata=extra_meta,
                )
            )

        self._current_plan = Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            steps=plan_steps,
            source=source,
        )
        return self._current_plan

    def update_step(
        self,
        *,
        step_index: int,
        status: str,
        result_summary: str | None = None,
        evidence: str | None = None,
        confidence: float | None = None,
        blocked_reason: str | None = None,
        required_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._current_plan is None:
            return
        if not (0 <= step_index < len(self._current_plan.steps)):
            return

        step = self._current_plan.steps[step_index]
        step.status = status  # type: ignore[assignment]

        if result_summary is not None:
            step.result_summary = result_summary
        if evidence:
            if evidence not in step.evidence:
                step.evidence.append(evidence)
        if confidence is not None:
            step.confidence = max(0.0, min(1.0, float(confidence)))
        if blocked_reason is not None:
            step.blocked_reason = blocked_reason
        if required_tools:
            merged = list(dict.fromkeys([*step.required_tools, *required_tools]))
            step.required_tools = merged
        if metadata:
            step.metadata.update(metadata)

        if status == "in_progress" and step_index > self._current_plan.current_step_index:
            self._current_plan.current_step_index = step_index
        elif status == "completed":
            next_idx = self._first_open_step_index()
            if next_idx is not None:
                self._current_plan.current_step_index = next_idx
            else:
                self._current_plan.current_step_index = max(0, len(self._current_plan.steps) - 1)

    def append_step(
        self,
        description: str,
        *,
        required_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        if self._current_plan is None:
            return None
        clean_description = description.strip()
        if not clean_description:
            return None
        step = PlanStep(
            id=str(uuid.uuid4()),
            description=clean_description,
            status="pending",
            required_tools=[tool.strip() for tool in (required_tools or []) if tool.strip()],
            metadata=metadata or {},
        )
        self._current_plan.steps.append(step)
        if self._current_plan.current_step is None:
            self._current_plan.current_step_index = len(self._current_plan.steps) - 1
        return len(self._current_plan.steps) - 1

    def next_pending_index(self) -> int | None:
        if self._current_plan is None:
            return None
        for idx, step in enumerate(self._current_plan.steps):
            if step.status in {"pending", "in_progress"}:
                return idx
        return None

    def _first_open_step_index(self) -> int | None:
        if self._current_plan is None:
            return None
        for idx, step in enumerate(self._current_plan.steps):
            if step.status not in {"completed", "skipped"}:
                return idx
        return None

    def format_current_plan(self) -> str | None:
        plan = self._current_plan
        if plan is None:
            return None
        lines = [f"Goal: {plan.goal}", plan.progress_summary]
        for idx, step in enumerate(plan.steps, start=1):
            marker = "->" if idx - 1 == plan.current_step_index else "  "
            lines.append(f"{marker} {idx}. [{step.status}] {step.description}")
            if step.required_tools:
                lines.append(f"   tools: {', '.join(step.required_tools)}")
            if step.result_summary:
                lines.append(f"   result: {step.result_summary}")
            if step.blocked_reason:
                lines.append(f"   blocked_reason: {step.blocked_reason}")
        return "\n".join(lines)

