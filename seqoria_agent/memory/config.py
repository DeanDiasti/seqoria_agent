from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ContextConfig:
    max_recent_messages: int = 20
    max_recent_tokens: int = 8000
    compression_enabled: bool = True
    compression_strategy: Literal["summarize_chunks", "progressive", "selective", "hierarchical"] = "hierarchical"
    compress_after_turns: int = 15
    compress_after_tokens: int = 6000
    checkpoint_every_n_turns: int | None = 5
    checkpoint_on_token_threshold: int | None = 4000
    checkpoint_on_semantic_shift: float | None = 0.3
    checkpoint_after_tool_chain: bool = True
    max_checkpoints: int = 10


@dataclass(slots=True)
class MemoryConfig:
    generate_hybrid: bool = True
    generate_full: bool = True
    retrieve_top_k: int = 5
    relevance_threshold: float = 0.6
    include_graph_neighbors: bool = True
    max_neighbor_depth: int = 1
    retrieval_recent_ratio: float = 0.6
    retrieval_long_term_ratio: float = 0.3
    retrieval_tool_ratio: float = 0.1
    capture_turn_memories: bool = True
    turn_memory_window: int = 6
    tool_memory_window: int = 5
    auto_select_resolution: bool = True
    default_resolution: Literal["compressed", "hybrid", "full"] = "compressed"


@dataclass(slots=True)
class PlanConfig:
    enabled: bool = True
    auto_infer: bool = True
    inject_in_context: bool = True
    update_on_tool_success: bool = True
    assistant_progress_enabled: bool = True
    replan_on_blocked: bool = True
    complexity_min_words: int = 18
    max_inferred_steps: int = 5
    min_progress_confidence: float = 0.55


@dataclass(slots=True)
class ToolHandlerConfig:
    store_successful_calls: bool = True
    include_surrounding_context: bool = True
    context_turns_before: int = 2
    context_turns_after: int = 1
    max_retries_before_rewrite: int = 2
    rewrite_failed_calls: bool = True
    soft_failure_message: bool = True


@dataclass(slots=True)
class AgenticMemoryConfig:
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    tool_handler: ToolHandlerConfig = field(default_factory=ToolHandlerConfig)
    memory_agent_provider: Any | None = None
    persist_memories: bool = False
    session_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    total_context_budget: int = 64000

