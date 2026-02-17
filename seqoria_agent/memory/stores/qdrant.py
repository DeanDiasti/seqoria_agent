from __future__ import annotations


from seqoria_agent.memory.models import MemoryScope
from seqoria_agent.memory.stores.base import VectorStoreProtocol


class QdrantVectorStore(VectorStoreProtocol):
    """Vector index adapter backed by Qdrant."""

    def __init__(
        self,
        *,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("QdrantVectorStore requires `qdrant-client` to be installed.") from exc

        self._qdrant_models = qdrant_models
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._ensure_collection_if_possible()

    def _scope_key(self, scope: MemoryScope) -> str:
        return scope.key()

    def _ensure_collection_if_possible(self) -> None:
        if self._vector_size is None:
            return
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self._collection_name in existing:
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=self._qdrant_models.VectorParams(
                size=self._vector_size,
                distance=self._qdrant_models.Distance.COSINE,
            ),
        )

    def _ensure_collection(self, embedding_size: int) -> None:
        if self._vector_size is None:
            self._vector_size = embedding_size
        if embedding_size != self._vector_size:
            raise ValueError(
                f"QdrantVectorStore expects vectors with {self._vector_size} dimensions, "
                f"received {embedding_size}."
            )
        self._ensure_collection_if_possible()

    def upsert(self, node_id: str, embedding: list[float], *, scope: MemoryScope) -> None:
        self._ensure_collection(len(embedding))
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                self._qdrant_models.PointStruct(
                    id=node_id,
                    vector=embedding,
                    payload={
                        "scope_key": self._scope_key(scope),
                        "node_id": node_id,
                    },
                )
            ],
        )

    def search(
        self,
        embedding: list[float],
        *,
        scope: MemoryScope,
        top_k: int,
    ) -> list[tuple[str, float]]:
        self._ensure_collection(len(embedding))
        query_filter = self._qdrant_models.Filter(
            must=[
                self._qdrant_models.FieldCondition(
                    key="scope_key",
                    match=self._qdrant_models.MatchValue(value=self._scope_key(scope)),
                )
            ]
        )
        if hasattr(self._client, "query_points"):
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=embedding,
                limit=max(1, top_k),
                with_payload=True,
                query_filter=query_filter,
            ).points
        else:  # pragma: no cover - backwards compatibility for older qdrant-client
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=embedding,
                limit=max(1, top_k),
                with_payload=True,
                query_filter=query_filter,
            )
        return [
            (
                str((point.payload or {}).get("node_id", point.id)),
                float(point.score if point.score is not None else 0.0),
            )
            for point in results
        ]

    def delete(self, node_id: str, *, scope: MemoryScope) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=self._qdrant_models.FilterSelector(
                filter=self._qdrant_models.Filter(
                    must=[
                        self._qdrant_models.FieldCondition(
                            key="scope_key",
                            match=self._qdrant_models.MatchValue(value=self._scope_key(scope)),
                        ),
                        self._qdrant_models.FieldCondition(
                            key="node_id",
                            match=self._qdrant_models.MatchValue(value=node_id),
                        ),
                    ]
                )
            ),
        )
