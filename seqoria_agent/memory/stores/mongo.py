from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from seqoria_agent.memory.models import Checkpoint, MemoryEdge, MemoryNode, MemoryScope, Plan
from seqoria_agent.memory.stores.base import MemoryRepositoryProtocol


class MongoMemoryRepository(MemoryRepositoryProtocol):
    """Canonical repository backed by MongoDB collections."""

    def __init__(
        self,
        *,
        uri: str = "mongodb://localhost:27017",
        database: str = "seqoria_agent_memory",
    ) -> None:
        try:
            from pymongo import ASCENDING, MongoClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("MongoMemoryRepository requires `pymongo` to be installed.") from exc

        self._ASCENDING = ASCENDING
        self._client = MongoClient(uri)
        self._db = self._client[database]
        self._nodes = self._db["memory_nodes"]
        self._edges = self._db["memory_edges"]
        self._checkpoints = self._db["memory_checkpoints"]
        self._events = self._db["memory_events"]
        self._plans = self._db["memory_plans"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._nodes.create_index(
            [("scope_key", self._ASCENDING), ("id", self._ASCENDING)],
            unique=True,
            name="scope_node_unique",
        )
        self._edges.create_index(
            [
                ("scope_key", self._ASCENDING),
                ("source_id", self._ASCENDING),
                ("target_id", self._ASCENDING),
                ("relationship", self._ASCENDING),
            ],
            unique=True,
            name="scope_edge_unique",
        )
        self._checkpoints.create_index(
            [("scope_key", self._ASCENDING), ("id", self._ASCENDING)],
            unique=True,
            name="scope_checkpoint_unique",
        )
        self._events.create_index(
            [("scope_key", self._ASCENDING), ("timestamp", self._ASCENDING)],
            name="scope_event_ts",
        )
        self._plans.create_index(
            [("scope_key", self._ASCENDING)],
            unique=True,
            name="scope_plan_unique",
        )

    def _scope_key(self, scope: MemoryScope) -> str:
        return scope.key()

    def _serialize(self, payload: dict[str, Any], scope: MemoryScope) -> dict[str, Any]:
        doc = dict(payload)
        doc["scope_key"] = self._scope_key(scope)
        return doc

    # Nodes -----------------------------------------------------------------
    def upsert_node(self, *, scope: MemoryScope, node: MemoryNode) -> None:
        payload = self._serialize(node.model_dump(mode="json"), scope)
        self._nodes.replace_one(
            {"scope_key": payload["scope_key"], "id": node.id},
            payload,
            upsert=True,
        )

    def get_node(self, *, scope: MemoryScope, node_id: str) -> MemoryNode | None:
        payload = self._nodes.find_one({"scope_key": self._scope_key(scope), "id": node_id})
        if not payload:
            return None
        payload.pop("_id", None)
        payload.pop("scope_key", None)
        return MemoryNode(**payload)

    def list_nodes(
        self,
        *,
        scope: MemoryScope,
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        query: dict[str, Any] = {"scope_key": self._scope_key(scope)}
        if memory_type is not None:
            query["memory_type"] = memory_type
        cursor = self._nodes.find(query).sort("created_at", self._ASCENDING)
        if limit is not None:
            cursor = cursor.limit(limit)
        nodes: list[MemoryNode] = []
        for payload in cursor:
            payload.pop("_id", None)
            payload.pop("scope_key", None)
            nodes.append(MemoryNode(**payload))
        return nodes

    def remove_node(self, *, scope: MemoryScope, node_id: str) -> None:
        scope_key = self._scope_key(scope)
        self._nodes.delete_one({"scope_key": scope_key, "id": node_id})
        self._edges.delete_many(
            {
                "scope_key": scope_key,
                "$or": [{"source_id": node_id}, {"target_id": node_id}],
            }
        )

    def set_node_index_state(
        self,
        *,
        scope: MemoryScope,
        node_id: str,
        status: Literal["pending", "indexed", "failed"],
        index_version: int | None = None,
    ) -> None:
        update: dict[str, Any] = {"index_status": status}
        if index_version is not None:
            update["index_version"] = index_version
        self._nodes.update_one(
            {"scope_key": self._scope_key(scope), "id": node_id},
            {"$set": update},
        )

    # Edges -----------------------------------------------------------------
    def upsert_edge(self, *, scope: MemoryScope, edge: MemoryEdge) -> None:
        payload = self._serialize(edge.model_dump(mode="json"), scope)
        self._edges.replace_one(
            {
                "scope_key": payload["scope_key"],
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship": edge.relationship,
            },
            payload,
            upsert=True,
        )

    def list_edges(self, *, scope: MemoryScope, node_id: str | None = None) -> list[MemoryEdge]:
        query: dict[str, Any] = {"scope_key": self._scope_key(scope)}
        if node_id is not None:
            query["$or"] = [{"source_id": node_id}, {"target_id": node_id}]
        edges: list[MemoryEdge] = []
        for payload in self._edges.find(query):
            payload.pop("_id", None)
            payload.pop("scope_key", None)
            edges.append(MemoryEdge(**payload))
        return edges

    # Checkpoints -----------------------------------------------------------
    def save_checkpoint(self, *, scope: MemoryScope, checkpoint: Checkpoint) -> None:
        payload = self._serialize(checkpoint.model_dump(mode="json"), scope)
        self._checkpoints.replace_one(
            {"scope_key": payload["scope_key"], "id": checkpoint.id},
            payload,
            upsert=True,
        )

    def list_checkpoints(self, *, scope: MemoryScope) -> list[Checkpoint]:
        query = {"scope_key": self._scope_key(scope)}
        results: list[Checkpoint] = []
        for payload in self._checkpoints.find(query).sort("created_at", self._ASCENDING):
            payload.pop("_id", None)
            payload.pop("scope_key", None)
            results.append(Checkpoint(**payload))
        return results

    # Timeline / events -----------------------------------------------------
    def append_event(self, *, scope: MemoryScope, event: dict[str, Any]) -> None:
        payload = self._serialize(dict(event), scope)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._events.insert_one(payload)

    def list_events(self, *, scope: MemoryScope, limit: int | None = None) -> list[dict[str, Any]]:
        cursor = self._events.find({"scope_key": self._scope_key(scope)}).sort(
            "timestamp",
            self._ASCENDING,
        )
        if limit is not None:
            cursor = cursor.limit(limit)
        events: list[dict[str, Any]] = []
        for payload in cursor:
            payload.pop("_id", None)
            payload.pop("scope_key", None)
            events.append(payload)
        return events

    # Plan ------------------------------------------------------------------
    def save_plan(self, *, scope: MemoryScope, plan: Plan | None) -> None:
        scope_key = self._scope_key(scope)
        if plan is None:
            self._plans.delete_one({"scope_key": scope_key})
            return
        payload = self._serialize(plan.model_dump(mode="json"), scope)
        self._plans.replace_one({"scope_key": scope_key}, payload, upsert=True)

    def get_plan(self, *, scope: MemoryScope) -> Plan | None:
        payload = self._plans.find_one({"scope_key": self._scope_key(scope)})
        if not payload:
            return None
        payload.pop("_id", None)
        payload.pop("scope_key", None)
        return Plan(**payload)
