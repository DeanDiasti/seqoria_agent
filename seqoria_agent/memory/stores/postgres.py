from __future__ import annotations

import json
from typing import Any, Literal

from seqoria_agent.memory.models import Checkpoint, MemoryEdge, MemoryNode, MemoryScope, Plan
from seqoria_agent.memory.stores.base import MemoryRepositoryProtocol


class PostgresMemoryRepository(MemoryRepositoryProtocol):
    """Canonical repository backed by PostgreSQL JSONB tables."""

    def __init__(self, *, dsn: str, schema: str = "public") -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PostgresMemoryRepository requires `psycopg` to be installed.") from exc

        self._psycopg = psycopg
        self._schema = schema
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema()

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("memory_records")} (
                    scope_key TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scope_key, record_type, record_id)
                );
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("memory_events")} (
                    id BIGSERIAL PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("memory_plans")} (
                    scope_key TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS memory_records_scope_type_idx ON {self._table('memory_records')} (scope_key, record_type);"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS memory_events_scope_idx ON {self._table('memory_events')} (scope_key, created_at);"
            )

    def _scope_key(self, scope: MemoryScope) -> str:
        return scope.key()

    def _upsert_record(
        self,
        *,
        scope: MemoryScope,
        record_type: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table("memory_records")} (scope_key, record_type, record_id, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (scope_key, record_type, record_id)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW();
                """,
                (self._scope_key(scope), record_type, record_id, json.dumps(payload)),
            )

    def _list_records(self, *, scope: MemoryScope, record_type: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT payload
                FROM {self._table("memory_records")}
                WHERE scope_key = %s AND record_type = %s
                ORDER BY updated_at ASC;
                """,
                (self._scope_key(scope), record_type),
            )
            return [row[0] for row in cursor.fetchall()]

    # Nodes -----------------------------------------------------------------
    def upsert_node(self, *, scope: MemoryScope, node: MemoryNode) -> None:
        self._upsert_record(
            scope=scope,
            record_type="node",
            record_id=node.id,
            payload=node.model_dump(mode="json"),
        )

    def get_node(self, *, scope: MemoryScope, node_id: str) -> MemoryNode | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT payload
                FROM {self._table("memory_records")}
                WHERE scope_key = %s AND record_type = 'node' AND record_id = %s;
                """,
                (self._scope_key(scope), node_id),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return MemoryNode(**row[0])

    def list_nodes(
        self,
        *,
        scope: MemoryScope,
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        rows = [MemoryNode(**payload) for payload in self._list_records(scope=scope, record_type="node")]
        if memory_type is not None:
            rows = [item for item in rows if item.memory_type == memory_type]
        rows.sort(key=lambda node: node.created_at)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def remove_node(self, *, scope: MemoryScope, node_id: str) -> None:
        scope_key = self._scope_key(scope)
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                DELETE FROM {self._table("memory_records")}
                WHERE scope_key = %s
                  AND (
                    (record_type = 'node' AND record_id = %s)
                    OR (
                        record_type = 'edge'
                        AND (
                            (payload->>'source_id') = %s
                            OR (payload->>'target_id') = %s
                        )
                    )
                  );
                """,
                (scope_key, node_id, node_id, node_id),
            )

    def set_node_index_state(
        self,
        *,
        scope: MemoryScope,
        node_id: str,
        status: Literal["pending", "indexed", "failed"],
        index_version: int | None = None,
    ) -> None:
        node = self.get_node(scope=scope, node_id=node_id)
        if node is None:
            return
        node.index_status = status
        if index_version is not None:
            node.index_version = index_version
        self.upsert_node(scope=scope, node=node)

    # Edges -----------------------------------------------------------------
    def upsert_edge(self, *, scope: MemoryScope, edge: MemoryEdge) -> None:
        record_id = f"{edge.source_id}:{edge.target_id}:{edge.relationship}"
        self._upsert_record(
            scope=scope,
            record_type="edge",
            record_id=record_id,
            payload=edge.model_dump(mode="json"),
        )

    def list_edges(self, *, scope: MemoryScope, node_id: str | None = None) -> list[MemoryEdge]:
        rows = [MemoryEdge(**payload) for payload in self._list_records(scope=scope, record_type="edge")]
        if node_id is None:
            return rows
        return [edge for edge in rows if edge.source_id == node_id or edge.target_id == node_id]

    # Checkpoints -----------------------------------------------------------
    def save_checkpoint(self, *, scope: MemoryScope, checkpoint: Checkpoint) -> None:
        self._upsert_record(
            scope=scope,
            record_type="checkpoint",
            record_id=checkpoint.id,
            payload=checkpoint.model_dump(mode="json"),
        )

    def list_checkpoints(self, *, scope: MemoryScope) -> list[Checkpoint]:
        rows = [Checkpoint(**payload) for payload in self._list_records(scope=scope, record_type="checkpoint")]
        rows.sort(key=lambda checkpoint: checkpoint.created_at)
        return rows

    # Timeline / events -----------------------------------------------------
    def append_event(self, *, scope: MemoryScope, event: dict[str, Any]) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table("memory_events")} (scope_key, payload)
                VALUES (%s, %s::jsonb);
                """,
                (self._scope_key(scope), json.dumps(event)),
            )

    def list_events(self, *, scope: MemoryScope, limit: int | None = None) -> list[dict[str, Any]]:
        sql = f"""
            SELECT payload
            FROM {self._table("memory_events")}
            WHERE scope_key = %s
            ORDER BY created_at ASC
        """
        params: list[Any] = [self._scope_key(scope)]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return [row[0] for row in cursor.fetchall()]

    # Plan ------------------------------------------------------------------
    def save_plan(self, *, scope: MemoryScope, plan: Plan | None) -> None:
        scope_key = self._scope_key(scope)
        with self._conn.cursor() as cursor:
            if plan is None:
                cursor.execute(
                    f"DELETE FROM {self._table('memory_plans')} WHERE scope_key = %s;",
                    (scope_key,),
                )
                return
            cursor.execute(
                f"""
                INSERT INTO {self._table("memory_plans")} (scope_key, payload)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (scope_key)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW();
                """,
                (scope_key, json.dumps(plan.model_dump(mode="json"))),
            )

    def get_plan(self, *, scope: MemoryScope) -> Plan | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self._table('memory_plans')} WHERE scope_key = %s;",
                (self._scope_key(scope),),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Plan(**row[0])
