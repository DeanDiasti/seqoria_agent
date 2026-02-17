from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryScope(BaseModel):
    """Isolation key used for memory retrieval and persistence."""

    model_config = ConfigDict(slots=True)

    tenant_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None

    def key(self) -> str:
        """Build a deterministic scope key."""
        return "|".join(
            [
                self.tenant_id or "",
                self.user_id or "",
                self.conversation_id or "",
                self.session_id or "",
            ]
        )

    def matches(self, other: "MemoryScope") -> bool:
        """Return True when two scopes are identical across populated fields."""
        return self.key() == other.key()


class MemoryNode(BaseModel):
    """A memory record with multiple text resolutions."""

    model_config = ConfigDict(slots=True)

    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    memory_type: Literal[
        "conversation",
        "tool_call",
        "tool_chain",
        "plan",
        "fact",
        "error",
    ] = "fact"
    compressed: str
    hybrid: str | None = None
    full: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: MemoryScope = Field(default_factory=MemoryScope)
    index_status: Literal["pending", "indexed", "failed"] = "indexed"
    index_version: int = 0
    last_accessed_at: datetime | None = None
    importance_score: float = 0.5


class MemoryEdge(BaseModel):
    """Graph edge connecting two memory nodes."""

    model_config = ConfigDict(slots=True)

    source_id: str
    target_id: str
    relationship: Literal[
        "leads_to",
        "caused_by",
        "related_to",
        "part_of",
        "contradicts",
        "refines",
        "same_session",
    ] = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    """A single plan step."""

    model_config = ConfigDict(slots=True)

    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed", "blocked", "skipped"] = "pending"
    result_summary: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    blocked_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Plan state for multi-step tasks."""

    model_config = ConfigDict(slots=True)

    id: str
    goal: str
    steps: list[PlanStep]
    current_step_index: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["inferred", "explicit"] = "explicit"

    @property
    def current_step(self) -> PlanStep | None:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress_summary(self) -> str:
        completed = sum(1 for step in self.steps if step.status == "completed")
        return f"Step {self.current_step_index + 1} of {len(self.steps)} ({completed} completed)"


class Checkpoint(BaseModel):
    """Checkpoint snapshot for restoration."""

    model_config = ConfigDict(slots=True)

    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_index: int
    messages_snapshot: list[dict[str, Any]]
    plan_snapshot: Plan | None = None
    trigger: Literal[
        "turn_count",
        "token_threshold",
        "semantic_shift",
        "tool_chain",
        "manual",
    ] = "manual"
    description: str | None = None

