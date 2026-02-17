from __future__ import annotations

from seqoria_agent.memory.models import MemoryScope
from seqoria_agent.memory.stores.base import VectorStoreProtocol


class PgVectorStore(VectorStoreProtocol):
    """Vector index adapter backed by PostgreSQL + pgvector."""

    def __init__(
        self,
        *,
        dsn: str,
        dimensions: int,
        schema: str = "public",
        table_name: str = "memory_vectors",
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PgVectorStore requires `psycopg` to be installed.") from exc
        try:
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PgVectorStore requires `pgvector` to be installed.") from exc

        self._psycopg = psycopg
        self._schema = schema
        self._table_name = table_name
        self._dimensions = dimensions
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._ensure_schema()

    def _table(self) -> str:
        return f"{self._schema}.{self._table_name}"

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table()} (
                    scope_key TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    embedding VECTOR({self._dimensions}) NOT NULL,
                    PRIMARY KEY (scope_key, node_id)
                );
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_scope_idx
                ON {self._table()} (scope_key);
                """
            )

    def upsert(self, node_id: str, embedding: list[float], *, scope: MemoryScope) -> None:
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Expected vector with {self._dimensions} dimensions, received {len(embedding)}"
            )
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table()} (scope_key, node_id, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (scope_key, node_id)
                DO UPDATE SET embedding = EXCLUDED.embedding;
                """,
                (scope.key(), node_id, embedding),
            )

    def search(
        self,
        embedding: list[float],
        *,
        scope: MemoryScope,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Expected vector with {self._dimensions} dimensions, received {len(embedding)}"
            )
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT node_id, 1 - (embedding <=> %s) AS similarity
                FROM {self._table()}
                WHERE scope_key = %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (embedding, scope.key(), embedding, max(top_k, 1)),
            )
            return [(row[0], float(row[1])) for row in cursor.fetchall()]

    def delete(self, node_id: str, *, scope: MemoryScope) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self._table()} WHERE scope_key = %s AND node_id = %s;",
                (scope.key(), node_id),
            )
