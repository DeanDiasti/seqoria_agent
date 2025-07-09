from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

__all__: list[str] = ["ToolInfo"]


class ToolInfo(BaseModel):
    """Metadata container for a registered XMCP tool."""

    func: Callable[..., Any]
    description: str | None = None
    # Names of parameters that are populated by the server (not the LLM)
    server_params: list[str] = []
    # Explicit DAG dependency declaration
    depends_on: list[str] = []
    # JSON-schema for the AI visible arguments
    input_schema: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True) 