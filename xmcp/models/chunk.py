"""Streaming chunk value object – used by :pymeth:`xmcp.client.core.AIClient.stream_chat`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal


@dataclass
class ChatChunk:  # noqa: D101 – simple container
    """Incremental piece of a streamed chat response.

    Attributes
    ----------
    type:
        Either ``"llm_delta"`` for partial LLM output or ``"tool_display"``
        when a *display tool* emits a :class:`xmcp.models.envelope.DisplayEnvelope`.
    text:
        Token delta from the LLM (only set when ``type == 'llm_delta'``).
    envelope:
        Serialized :class:`xmcp.models.envelope.DisplayEnvelope` when
        ``type == 'tool_display'``.
    """

    type: Literal["llm_delta", "tool_display"]
    text: Optional[str] = None
    envelope: Optional[Dict] = None 