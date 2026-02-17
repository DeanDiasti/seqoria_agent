from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seqoria_agent.memory.models import MemoryScope
from seqoria_agent.memory.stores.base import PersistenceStoreProtocol


class FilePersistenceStore(PersistenceStoreProtocol):
    """Persist memory state as JSON files on local disk."""

    def __init__(self, base_dir: str = "memories") -> None:
        self.base_dir = Path(base_dir)

    async def save_state(
        self,
        *,
        scope: MemoryScope,
        state: dict[str, Any],
        path: str | None = None,
    ) -> str:
        file_path = Path(path) if path else self._path_for_scope(scope)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return str(file_path)

    async def load_state(
        self,
        *,
        scope: MemoryScope,
        path: str | None = None,
    ) -> dict[str, Any] | None:
        file_path = Path(path) if path else self._path_for_scope(scope)
        if not file_path.exists():
            return None
        return json.loads(file_path.read_text(encoding="utf-8"))

    def _path_for_scope(self, scope: MemoryScope) -> Path:
        safe_key = scope.key().replace("|", "__") or "default"
        return self.base_dir / f"{safe_key}.json"

