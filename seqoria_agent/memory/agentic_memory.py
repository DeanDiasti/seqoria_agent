from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from seqoria_agent.memory.agent.memory_agent import InferredPlan, MemoryAgent, PlanProgressUpdate
from seqoria_agent.memory.background import BackgroundWorker
from seqoria_agent.memory.config import (
    AgenticMemoryConfig,
    ContextConfig,
    MemoryConfig,
    PlanConfig,
    ToolHandlerConfig,
)
from seqoria_agent.memory.context import ContextAssembler, ContextManager, Compressor, estimate_tokens, message_to_dict
from seqoria_agent.memory.embeddings.base import EmbedderProtocol
from seqoria_agent.memory.graph.store import MemoryGraph
from seqoria_agent.memory.managers.plan import PlanManager
from seqoria_agent.memory.managers.tool_handler import ToolCallHandler
from seqoria_agent.memory.models import Checkpoint, MemoryNode, MemoryScope, Plan
from seqoria_agent.memory.stores.base import (
    MemoryRepositoryProtocol,
    PersistenceStoreProtocol,
    VectorStoreProtocol,
)
from seqoria_agent.memory.stores.in_memory import InMemoryMemoryRepository, InMemoryVectorStore
from seqoria_agent.memory.stores.persistent import FilePersistenceStore
from seqoria_agent.models.chunk import (
    AssistantMessage,
    LLMUsage,
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
    UserMessage,
)


class AgenticMemory:
    """Intelligent memory orchestrator compatible with list-like history flows."""

    def __init__(
        self,
        config: AgenticMemoryConfig | None = None,
        *,
        vector_store: VectorStoreProtocol | None = None,
        memory_repository: MemoryRepositoryProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        memory_agent_provider: Any | None = None,
        persistence_store: PersistenceStoreProtocol | None = None,
    ) -> None:
        self.config = config or AgenticMemoryConfig()
        self._scope = MemoryScope(
            tenant_id=self.config.tenant_id,
            user_id=self.config.user_id,
            conversation_id=self.config.conversation_id,
            session_id=self.config.session_id,
        )
        self._embedder = embedder
        self._memory_agent = MemoryAgent()
        self._memory_agent_provider = memory_agent_provider or self.config.memory_agent_provider
        self._context_manager = ContextManager(self.config.context)
        self._assembler = ContextAssembler()
        self._compressor = Compressor(strategy=self.config.context.compression_strategy)
        self._repository = memory_repository or InMemoryMemoryRepository()
        self._graph = MemoryGraph(
            vector_store=vector_store or InMemoryVectorStore(),
            repository=self._repository,
        )
        self._plan_manager = PlanManager()
        self._tool_handler = ToolCallHandler(self.config.tool_handler)
        self._background = BackgroundWorker()
        self._checkpoints: list[Checkpoint] = self._repository.list_checkpoints(scope=self._scope)
        self._turn_count = 0
        self._current_retrieval_query: str | None = None
        self._persistence = persistence_store or FilePersistenceStore()
        self._last_turn_memory_signature: str | None = None
        self._index_version_counter = 0
        self._applied_plan_signatures: set[str] = set()
        existing_plan = self._repository.get_plan(scope=self._scope)
        if existing_plan is not None:
            self._plan_manager._current_plan = existing_plan

    # ------------------------------------------------------------------
    # List-like behavior
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.to_messages())

    def __len__(self) -> int:
        return len(self.to_messages())

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._context_manager.raw_messages[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._context_manager.set_message(index, value)

    def __delitem__(self, index: int) -> None:
        self._context_manager.delete_message(index)

    def append(self, item: Any) -> None:
        self._append_normalized(item)

    def extend(self, items: Iterable[Any]) -> None:
        for item in items:
            self.append(item)

    def insert(self, index: int, item: Any) -> None:
        self._context_manager.insert_message(index, item)

    # ------------------------------------------------------------------
    # Turn lifecycle hooks (memory-first orchestration)
    # ------------------------------------------------------------------
    def begin_turn(
        self,
        user_query: str,
        *,
        retrieval_query: str | None = None,
    ) -> list[dict[str, Any]]:
        self.add_user_message(user_query)
        self._current_retrieval_query = retrieval_query or user_query
        self._maybe_infer_plan(user_query)
        return self.messages_for_llm(query=self._current_retrieval_query)

    def messages_for_llm(self, query: str | None = None) -> list[dict[str, Any]]:
        return self.to_messages(query=query)

    def record_tool_cycle(
        self,
        tool_calls: list[ToolCallRequest],
        tool_results: list[ToolCallResult | ToolCallError],
    ) -> None:
        for req, result in zip(tool_calls, tool_results):
            self.add_tool_call(req)
            if isinstance(result, ToolCallError):
                self.add_tool_result(
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                    result=result.error,
                    success=False,
                )
            else:
                self.add_tool_result(
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                    result=str(result.result),
                    success=True,
                )
        self._maybe_update_plan_from_tools(tool_calls, tool_results)

    def commit_assistant_response(
        self,
        content: str,
        *,
        model: str | None = None,
        usage: LLMUsage | None = None,
    ) -> None:
        self.add_assistant_message(content)
        if self._context_manager.raw_messages:
            payload = self._context_manager.raw_messages[-1]
            if model is not None:
                payload["model"] = model
            if usage is not None:
                payload["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        self._maybe_update_plan_from_response(content)

    def end_turn(self) -> None:
        self._turn_count += 1
        if self._check_checkpoint_triggers():
            self.create_checkpoint(description="automatic checkpoint")
        self._maybe_compress_history()
        self._capture_turn_window_memory()
        self._append_timeline_event("turn_completed", {"turn_count": self._turn_count})
        self._current_retrieval_query = None

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------
    def add_user_message(self, content: str) -> None:
        self._context_manager.add_message(UserMessage(role="user", content=content))
        self._append_timeline_event("user_message", {"content": content})

    def add_assistant_message(self, content: str) -> None:
        self._context_manager.add_message(AssistantMessage(role="assistant", content=content))
        self._append_timeline_event("assistant_message", {"content": content})

    def add_system_message(self, content: str) -> None:
        self._context_manager.add_message({"role": "system", "content": content})
        self._append_timeline_event("system_message", {"content": content})

    @property
    def system_prompt(self) -> str | None:
        """Current base system prompt used during LLM context assembly."""
        return self._context_manager.base_system_prompt

    def update_system_prompt(self, content: str | None) -> None:
        """Update or clear the base system prompt without appending a new message."""
        self._context_manager.set_base_system_prompt(content)
        self._append_timeline_event(
            "system_prompt_updated",
            {
                "content": self._context_manager.base_system_prompt,
                "cleared": self._context_manager.base_system_prompt is None,
            },
        )

    def add_tool_call(self, tool_call: ToolCallRequest) -> None:
        self._context_manager.add_message(tool_call)
        payload = tool_call.model_dump() if hasattr(tool_call, "model_dump") else message_to_dict(tool_call)
        self._append_timeline_event("tool_call", payload)

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        result: str,
        success: bool = True,
    ) -> None:
        context_window = self._build_context_window_text(
            max_messages=self.config.memory.tool_memory_window,
        )
        if success:
            payload = ToolCallResult(
                tool_call_id=tool_call_id,
                role="tool_result",
                name=name,
                type="tool_use",
                result=result,
            )
            self._context_manager.add_message(payload)
            self._tool_handler.record_success(
                tool_call_id=tool_call_id,
                tool_name=name,
                result=result,
            )
            self._append_timeline_event(
                "tool_result",
                {
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "success": True,
                    "result": result,
                },
            )
            self._schedule_memory_capture(
                content=(
                    f"Tool outcome\n"
                    f"- status: success\n"
                    f"- tool: {name}\n"
                    f"- result: {result}\n"
                    f"- surrounding context:\n{context_window}"
                ),
                memory_type="tool_call",
                metadata={
                    "tool_name": name,
                    "success": True,
                    "tool_call_id": tool_call_id,
                    "source": "tool_outcome",
                    "context_window_messages": self.config.memory.tool_memory_window,
                },
            )
            return

        payload = ToolCallError(
            tool_call_id=tool_call_id,
            role="tool_error",
            name=name,
            type="tool_error",
            error=result,
        )
        self._context_manager.add_message(payload)
        guidance = self._tool_handler.record_failure(
            tool_call_id=tool_call_id,
            tool_name=name,
            error=result,
            retry_guidance=None,
        )
        self._append_timeline_event(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "success": False,
                "error": result,
            },
        )
        if guidance:
            self._context_manager.inject_system_note(guidance)
        self._schedule_memory_capture(
            content=(
                f"Tool outcome\n"
                f"- status: failure\n"
                f"- tool: {name}\n"
                f"- error: {result}\n"
                f"- retry guidance: {guidance or 'n/a'}\n"
                f"- surrounding context:\n{context_window}"
            ),
            memory_type="error",
            metadata={
                "tool_name": name,
                "success": False,
                "tool_call_id": tool_call_id,
                "source": "tool_outcome",
                "context_window_messages": self.config.memory.tool_memory_window,
            },
        )

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------
    def set_plan(
        self,
        goal: str,
        steps: list[str],
        *,
        source: str = "explicit",
        step_metadata: list[dict[str, Any]] | None = None,
    ) -> Plan:
        normalized_source = source if source in {"explicit", "inferred"} else "explicit"
        plan = self._plan_manager.set_plan(
            goal,
            steps,
            source=normalized_source,  # type: ignore[arg-type]
            step_metadata=step_metadata,
        )
        self._repository.save_plan(scope=self._scope, plan=plan)
        self._append_timeline_event(
            "plan_set",
            {"goal": goal, "steps": steps, "source": normalized_source},
        )
        return plan

    def update_plan_step(
        self,
        step_index: int,
        status: str,
        result_summary: str | None = None,
        *,
        evidence: str | None = None,
        confidence: float | None = None,
        blocked_reason: str | None = None,
        required_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._plan_manager.update_step(
            step_index=step_index,
            status=status,
            result_summary=result_summary,
            evidence=evidence,
            confidence=confidence,
            blocked_reason=blocked_reason,
            required_tools=required_tools,
            metadata=metadata,
        )
        self._repository.save_plan(scope=self._scope, plan=self._plan_manager.current_plan)
        self._append_timeline_event(
            "plan_step_update",
            {
                "step_index": step_index,
                "status": status,
                "result_summary": result_summary,
                "evidence": evidence,
                "confidence": confidence,
                "blocked_reason": blocked_reason,
                "required_tools": required_tools,
            },
        )

    @property
    def current_plan(self) -> Plan | None:
        return self._plan_manager.current_plan

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def create_checkpoint(self, description: str | None = None) -> Checkpoint:
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            turn_index=self._turn_count,
            messages_snapshot=copy.deepcopy(self._context_manager.raw_messages),
            plan_snapshot=copy.deepcopy(self._plan_manager.current_plan),
            trigger="manual" if description else "turn_count",
            description=description,
        )
        self._checkpoints.append(checkpoint)
        self._repository.save_checkpoint(scope=self._scope, checkpoint=checkpoint)
        self._append_timeline_event(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint.id,
                "turn_index": checkpoint.turn_index,
                "trigger": checkpoint.trigger,
                "description": checkpoint.description,
            },
        )
        if len(self._checkpoints) > self.config.context.max_checkpoints:
            self._checkpoints = self._checkpoints[-self.config.context.max_checkpoints :]
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> None:
        if not self._checkpoints:
            self._checkpoints = self._repository.list_checkpoints(scope=self._scope)
        found = next((cp for cp in self._checkpoints if cp.id == checkpoint_id), None)
        if found is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        self._context_manager.raw_messages = copy.deepcopy(found.messages_snapshot)
        self._plan_manager._current_plan = copy.deepcopy(found.plan_snapshot)
        self._turn_count = found.turn_index

    def list_checkpoints(self) -> list[Checkpoint]:
        if not self._checkpoints:
            self._checkpoints = self._repository.list_checkpoints(scope=self._scope)
        return list(self._checkpoints)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    async def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        metadata: dict | None = None,
    ) -> MemoryNode:
        metadata = metadata or {}
        compressed, hybrid, full = await self._memory_agent.generate_resolutions(
            content,
            memory_type=memory_type,
            metadata=metadata,
            generate_hybrid=self.config.memory.generate_hybrid,
            generate_full=self.config.memory.generate_full,
        )
        node = MemoryNode(
            id=str(uuid.uuid4()),
            memory_type=memory_type,  # type: ignore[arg-type]
            compressed=compressed,
            hybrid=hybrid,
            full=full,
            metadata=metadata,
            scope=self._scope,
            index_status="pending" if self._embedder else "indexed",
            index_version=0,
        )
        self._graph.add_node(node)
        if self._embedder is not None:
            self._schedule_indexing(node)
        else:
            self._graph.set_node_index_state(
                scope=self._scope,
                node_id=node.id,
                status="indexed",
                index_version=node.index_version,
            )
        self._connect_recent_memory(node)
        self._append_timeline_event(
            "memory_captured",
            {
                "memory_id": node.id,
                "memory_type": memory_type,
                "index_status": node.index_status,
            },
        )
        return node

    async def search_memories(
        self,
        query: str,
        top_k: int | None = None,
        memory_type: str | None = None,
    ) -> list[MemoryNode]:
        limit = top_k or self.config.memory.retrieve_top_k
        scored = await self._graph.search(
            query,
            scope=self._scope,
            top_k=limit,
            memory_type=memory_type,
            threshold=self.config.memory.relevance_threshold,
            embedder=self._embedder,
            include_neighbors=self.config.memory.include_graph_neighbors,
            neighbor_depth=self.config.memory.max_neighbor_depth,
        )
        return [node for node, _ in scored]

    def get_memory(self, memory_id: str) -> MemoryNode | None:
        node = self._graph.get_node(memory_id, scope=self._scope)
        if node and node.scope.matches(self._scope):
            return node
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        await self._background.start()
        self._resume_pending_index_jobs()

    async def stop(self) -> None:
        if self._background.is_running:
            await self._background.flush()
        await self._background.stop()

    @property
    def is_processing(self) -> bool:
        return self._background.is_processing

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        nodes = self._graph.list_nodes(scope=self._scope)
        edges = self._graph.list_edges(scope=self._scope)
        return {
            "config": asdict(self.config),
            "scope": self._scope.model_dump(),
            "turn_count": self._turn_count,
            "context": {
                "raw_messages": copy.deepcopy(self._context_manager.raw_messages),
                "compressed_history": self._context_manager.compressed_history,
                "base_system_prompt": self._context_manager.base_system_prompt,
            },
            "graph": {
                "nodes": [node.model_dump(mode="json") for node in nodes],
                "edges": [edge.model_dump(mode="json") for edge in edges],
            },
            "checkpoints": [checkpoint.model_dump(mode="json") for checkpoint in self._checkpoints],
            "plan": self._plan_manager.current_plan.model_dump(mode="json")
            if self._plan_manager.current_plan
            else None,
            "timeline": self._repository.list_events(scope=self._scope),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "AgenticMemory":
        if "config" not in kwargs:
            cfg_dict = data.get("config", {})
            kwargs["config"] = AgenticMemoryConfig(
                context=ContextConfig(**cfg_dict.get("context", {})),
                memory=MemoryConfig(**cfg_dict.get("memory", {})),
                plan=PlanConfig(**cfg_dict.get("plan", {})),
                tool_handler=ToolHandlerConfig(**cfg_dict.get("tool_handler", {})),
                memory_agent_provider=cfg_dict.get("memory_agent_provider"),
                persist_memories=cfg_dict.get("persist_memories", False),
                session_id=cfg_dict.get("session_id"),
                tenant_id=cfg_dict.get("tenant_id"),
                user_id=cfg_dict.get("user_id"),
                conversation_id=cfg_dict.get("conversation_id"),
                total_context_budget=cfg_dict.get("total_context_budget", 16000),
            )
        obj = cls(**kwargs)
        obj._scope = MemoryScope(**data.get("scope", {}))
        obj._turn_count = int(data.get("turn_count", 0))
        context = data.get("context", {})
        obj._context_manager.raw_messages = [message_to_dict(msg) for msg in context.get("raw_messages", [])]
        obj._context_manager.set_compressed_history(context.get("compressed_history"))
        obj._context_manager._base_system_prompt = context.get("base_system_prompt")  # noqa: SLF001
        max_index_version = 0
        for raw_node in data.get("graph", {}).get("nodes", []):
            node = MemoryNode(**raw_node)
            obj._graph.add_node(node)
            max_index_version = max(max_index_version, node.index_version)
        for raw_edge in data.get("graph", {}).get("edges", []):
            from seqoria_agent.memory.models import MemoryEdge

            obj._graph.add_edge(MemoryEdge(**raw_edge), scope=obj._scope)
        obj._checkpoints = [Checkpoint(**item) for item in data.get("checkpoints", [])]
        for checkpoint in obj._checkpoints:
            obj._repository.save_checkpoint(scope=obj._scope, checkpoint=checkpoint)
        if data.get("plan"):
            obj._plan_manager._current_plan = Plan(**data["plan"])
            obj._repository.save_plan(scope=obj._scope, plan=obj._plan_manager.current_plan)
        for event in data.get("timeline", []):
            obj._repository.append_event(scope=obj._scope, event=event)
        obj._index_version_counter = max_index_version
        return obj

    async def save(self, path: str | None = None) -> str:
        state = self.to_dict()
        if path is None and not self.config.persist_memories:
            raise ValueError("save(path=None) requires config.persist_memories=True")
        return await self._persistence.save_state(scope=self._scope, state=state, path=path)

    @classmethod
    async def load(
        cls,
        path: str,
        **kwargs: Any,
    ) -> "AgenticMemory":
        file_path = Path(path)
        if not file_path.exists():
            return cls(**kwargs)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, **kwargs)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------
    def to_messages(self, query: str | None = None) -> list[dict]:
        retrieval_query = query or self._current_retrieval_query or self._context_manager.latest_user_text()
        memory_text = self._format_retrieved_memories(retrieval_query) if retrieval_query else None
        plan_text = self._plan_manager.format_current_plan() if self.config.plan.inject_in_context else None
        tool_guidance = self._tool_handler.get_guidance_text()
        notes = self._context_manager.consume_injected_notes()
        recent_messages = self._context_manager.get_recent_messages(
            max_tokens=self.config.context.max_recent_tokens
        )
        return self._assembler.assemble(
            base_system_prompt=self._context_manager.base_system_prompt,
            memory_section=memory_text,
            plan_section=plan_text,
            tool_guidance_section=tool_guidance,
            compressed_history=self._context_manager.compressed_history,
            injected_notes=notes,
            recent_messages=recent_messages,
            total_budget=self.config.total_context_budget,
            token_estimator=estimate_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_timeline_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._repository.append_event(
            scope=self._scope,
            event={
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            },
        )

    def _next_index_version(self) -> int:
        self._index_version_counter += 1
        return self._index_version_counter

    def _resume_pending_index_jobs(self) -> None:
        if self._embedder is None:
            return
        for node in self._graph.list_nodes(scope=self._scope):
            if node.index_status == "pending":
                self._schedule_indexing(node)

    def _schedule_indexing(self, node: MemoryNode) -> None:
        if self._embedder is None:
            return

        async def _index_task() -> None:
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    await self._graph.ensure_embedding(node, embedder=self._embedder)
                    self._graph.set_node_index_state(
                        scope=self._scope,
                        node_id=node.id,
                        status="indexed",
                        index_version=self._next_index_version(),
                    )
                    self._append_timeline_event(
                        "memory_indexed",
                        {"memory_id": node.id, "attempt": attempt},
                    )
                    return
                except Exception as exc:  # pragma: no cover - defensive for provider failures
                    if attempt == max_attempts:
                        self._graph.set_node_index_state(
                            scope=self._scope,
                            node_id=node.id,
                            status="failed",
                        )
                        self._append_timeline_event(
                            "memory_index_failed",
                            {"memory_id": node.id, "error": str(exc)},
                        )

        if self._background.is_running:
            self._background.submit(_index_task)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(_index_task())

    def _maybe_infer_plan(self, user_query: str) -> None:
        cfg = self.config.plan
        if not cfg.enabled or not cfg.auto_infer:
            return
        if self._plan_manager.current_plan is not None:
            return

        inferred = self._memory_agent.infer_plan_from_turn(
            user_query=user_query,
            existing_plan=self._plan_manager.current_plan,
            complexity_min_words=cfg.complexity_min_words,
            max_steps=cfg.max_inferred_steps,
        )
        if inferred is None:
            return
        if inferred.confidence < cfg.min_progress_confidence:
            return

        plan = self.set_plan(
            inferred.goal,
            inferred.steps,
            source="inferred",
            step_metadata=inferred.step_metadata,
        )
        self._append_timeline_event(
            "plan_auto_inferred",
            {
                "plan_id": plan.id,
                "confidence": inferred.confidence,
                "rationale": inferred.rationale,
                "step_count": len(plan.steps),
            },
        )
        self._schedule_memory_capture(
            content=(
                "Auto-inferred execution plan\n"
                f"- goal: {plan.goal}\n"
                f"- steps: {len(plan.steps)}\n"
                f"- confidence: {inferred.confidence:.2f}\n"
                f"- rationale: {inferred.rationale}"
            ),
            memory_type="plan",
            metadata={
                "source": "auto_infer",
                "confidence": round(inferred.confidence, 3),
            },
        )

    def _maybe_update_plan_from_tools(
        self,
        tool_calls: list[ToolCallRequest],
        tool_results: list[ToolCallResult | ToolCallError],
    ) -> None:
        cfg = self.config.plan
        plan = self._plan_manager.current_plan
        if not cfg.enabled or not cfg.update_on_tool_success or plan is None:
            return
        progress = self._memory_agent.infer_step_progress(
            plan=plan,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        self._apply_inferred_plan_progress(progress, origin="tool_cycle")

    def _maybe_update_plan_from_response(self, content: str) -> None:
        cfg = self.config.plan
        plan = self._plan_manager.current_plan
        if not cfg.enabled or not cfg.assistant_progress_enabled or plan is None:
            return
        progress = self._memory_agent.infer_step_progress(
            plan=plan,
            assistant_text=content,
        )
        self._apply_inferred_plan_progress(progress, origin="assistant_response")

    def _apply_inferred_plan_progress(
        self,
        progress: PlanProgressUpdate | None,
        *,
        origin: str,
    ) -> None:
        if progress is None:
            return
        cfg = self.config.plan
        plan = self._plan_manager.current_plan
        if plan is None:
            return
        if progress.confidence < cfg.min_progress_confidence:
            return

        target_index = progress.step_index
        if target_index is None:
            target_index = self._plan_manager.next_pending_index()
        if target_index is None:
            return
        if not (0 <= target_index < len(plan.steps)):
            return
        target_step = plan.steps[target_index]
        if target_step.status in {"completed", "skipped"} and progress.status in {"completed", "in_progress"}:
            return

        signature = (
            f"{origin}|{target_index}|{progress.status}|"
            f"{self._normalize_signature_text(progress.evidence)}"
        )
        if not self._register_plan_signature(signature):
            return

        self.update_plan_step(
            target_index,
            progress.status,
            progress.result_summary,
            evidence=progress.evidence,
            confidence=progress.confidence,
            blocked_reason=progress.blocked_reason,
            required_tools=progress.required_tools,
            metadata={"source": origin},
        )
        self._append_timeline_event(
            "plan_step_auto_update",
            {
                "origin": origin,
                "step_index": target_index,
                "status": progress.status,
                "confidence": progress.confidence,
            },
        )
        self._schedule_memory_capture(
            content=(
                "Plan step auto-update\n"
                f"- origin: {origin}\n"
                f"- step: {target_index + 1}\n"
                f"- status: {progress.status}\n"
                f"- summary: {progress.result_summary}\n"
                f"- evidence: {progress.evidence}"
            ),
            memory_type="plan",
            metadata={
                "source": "auto_progress",
                "origin": origin,
                "step_index": target_index,
                "status": progress.status,
                "confidence": round(progress.confidence, 3),
            },
        )
        if progress.status in {"blocked", "failed"}:
            self._maybe_replan_from_blocker(target_index, progress)

    def _maybe_replan_from_blocker(self, step_index: int, progress: PlanProgressUpdate) -> None:
        cfg = self.config.plan
        plan = self._plan_manager.current_plan
        if plan is None or not cfg.replan_on_blocked:
            return

        reason = progress.blocked_reason or progress.evidence
        if not reason:
            return
        replan_signature = f"replan|{step_index}|{self._normalize_signature_text(reason)}"
        if not self._register_plan_signature(replan_signature):
            return

        recovery_step = self._memory_agent.replan_if_blocked(
            plan=plan,
            blocked_step_index=step_index,
            blocked_reason=reason,
            max_steps=cfg.max_inferred_steps,
        )
        if not recovery_step:
            return

        added_index = self._plan_manager.append_step(
            recovery_step,
            required_tools=progress.required_tools,
            metadata={"source": "auto_replan", "blocked_step_index": step_index},
        )
        if added_index is None:
            return
        self._repository.save_plan(scope=self._scope, plan=self._plan_manager.current_plan)
        self._append_timeline_event(
            "plan_auto_replan",
            {
                "blocked_step_index": step_index,
                "added_step_index": added_index,
                "description": recovery_step,
            },
        )

    def _normalize_signature_text(self, value: str) -> str:
        return " ".join(value.lower().split())[:200]

    def _register_plan_signature(self, signature: str) -> bool:
        if signature in self._applied_plan_signatures:
            return False
        self._applied_plan_signatures.add(signature)
        if len(self._applied_plan_signatures) > 800:
            self._applied_plan_signatures.clear()
        return True

    def _append_normalized(self, item: Any) -> None:
        data = message_to_dict(item)
        role = data.get("role")
        if role == "user":
            content = str(data.get("content", ""))
            self.add_user_message(content)
            return
        if role == "assistant":
            content = str(data.get("content", ""))
            self.add_assistant_message(content)
            return
        if role == "system":
            self.add_system_message(str(data.get("content", "")))
            return
        if role == "tool_call":
            self._context_manager.add_message(data)
            return
        if role in {"tool_result", "tool"}:
            self.add_tool_result(
                tool_call_id=str(data.get("tool_call_id", "")),
                name=str(data.get("name", "")),
                result=str(data.get("result", data.get("content", ""))),
                success=True,
            )
            return
        if role == "tool_error":
            self.add_tool_result(
                tool_call_id=str(data.get("tool_call_id", "")),
                name=str(data.get("name", "")),
                result=str(data.get("error", data.get("result", ""))),
                success=False,
            )
            return
        self._context_manager.add_message(data)

    def _check_checkpoint_triggers(self) -> bool:
        cfg = self.config.context
        if cfg.checkpoint_every_n_turns and self._turn_count > 0:
            if self._turn_count % cfg.checkpoint_every_n_turns == 0:
                return True
        if cfg.checkpoint_on_token_threshold:
            if self._context_manager.total_tokens > cfg.checkpoint_on_token_threshold:
                return True
        return False

    def _maybe_compress_history(self) -> None:
        cfg = self.config.context
        if not cfg.compression_enabled:
            return
        should_compress = False
        if cfg.compress_after_turns and self._turn_count >= cfg.compress_after_turns:
            should_compress = True
        if cfg.compress_after_tokens and self._context_manager.total_tokens >= cfg.compress_after_tokens:
            should_compress = True
        if not should_compress:
            return

        older = self._context_manager.older_messages_for_compression()
        summary = self._compressor.compress(
            older,
            existing_summary=self._context_manager.compressed_history,
        )
        self._context_manager.set_compressed_history(summary)
        if summary:
            self._schedule_memory_capture(
                content=summary,
                memory_type="conversation",
                metadata={"source": "compression", "turn": self._turn_count},
            )

    def _format_retrieved_memories(self, query: str) -> str | None:
        if not query.strip():
            return None
        scored = self._graph.search_lexical(
            query,
            scope=self._scope,
            top_k=self.config.memory.retrieve_top_k,
            resolution=self.config.memory.default_resolution,
        )
        if not scored:
            return None
        lines: list[str] = []
        for node, score in scored:
            selected = self._select_resolution(node, score=score)
            lines.append(f"- ({node.memory_type}, score={score:.2f}) {selected}")
        return "\n".join(lines)

    def _select_resolution(self, node: MemoryNode, *, score: float) -> str:
        if not self.config.memory.auto_select_resolution:
            default = self.config.memory.default_resolution
            if default == "full" and node.full:
                return node.full
            if default == "hybrid" and node.hybrid:
                return node.hybrid
            return node.compressed
        if score > 0.8 and node.full:
            return node.full
        if score > 0.7 and node.hybrid:
            return node.hybrid
        return node.compressed

    def _schedule_memory_capture(
        self,
        *,
        content: str,
        memory_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async def _task() -> None:
            await self.add_memory(content=content, memory_type=memory_type, metadata=metadata)

        if self._background.is_running:
            self._background.submit(_task)
            return

        # If no background worker is running, schedule best-effort task.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(_task())

    def _connect_recent_memory(self, node: MemoryNode) -> None:
        # Connect to most recent prior node in same scope for lightweight graph continuity.
        same_scope = [candidate for candidate in self._graph.list_nodes(scope=self._scope) if candidate.id != node.id]
        if not same_scope:
            return
        source = same_scope[-1]
        from seqoria_agent.memory.models import MemoryEdge

        edge = MemoryEdge(
            source_id=source.id,
            target_id=node.id,
            relationship="leads_to",
            weight=0.5,
            metadata={"source": "sequential"},
        )
        self._graph.add_edge(edge, scope=self._scope)

    def _build_context_window_text(self, *, max_messages: int) -> str:
        window = [
            message
            for message in self._context_manager.raw_messages
            if message.get("role") != "system"
        ][-max(1, max_messages) :]
        if not window:
            return "- n/a"
        return "\n".join(
            f"- {msg.get('role', 'unknown')}: {self._normalize_message_content(msg)[:220]}"
            for msg in window
        )

    def _normalize_message_content(self, message: dict[str, Any]) -> str:
        value = message.get("content")
        if isinstance(value, str):
            return " ".join(value.split())
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        chunks.append(str(item.get("text", "")))
                    elif "content" in item:
                        chunks.append(str(item.get("content")))
                    else:
                        chunks.append(str(item))
                else:
                    chunks.append(str(item))
            return " ".join(" ".join(chunks).split())
        if value is None:
            if "result" in message:
                return " ".join(str(message.get("result", "")).split())
            if "error" in message:
                return " ".join(str(message.get("error", "")).split())
        return " ".join(str(value).split())

    def _capture_turn_window_memory(self) -> None:
        if not self.config.memory.capture_turn_memories:
            return
        context_text = self._build_context_window_text(
            max_messages=self.config.memory.turn_memory_window,
        )
        if context_text == "- n/a":
            return
        if "- user:" not in context_text or "- assistant:" not in context_text:
            return
        signature = context_text[-300:]
        if signature == self._last_turn_memory_signature:
            return
        self._last_turn_memory_signature = signature
        self._schedule_memory_capture(
            content=(
                "Conversation window summary candidate\n"
                f"- turn: {self._turn_count}\n"
                f"- scope: {self._scope.key()}\n"
                f"- context:\n{context_text}"
            ),
            memory_type="conversation",
            metadata={
                "source": "turn_window",
                "turn": self._turn_count,
                "scope": self._scope.key(),
                "context_window_messages": self.config.memory.turn_memory_window,
            },
        )

