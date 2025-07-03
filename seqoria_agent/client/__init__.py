"""Client-side SDK for interacting with seqoria_agent servers."""

from seqoria_agent.models.chunk import ToolCallRequest, ToolCallResult  # noqa: F401

from seqoria_agent.client.ai_client.base import AIClient  # noqa: F401
from seqoria_agent.client.simple_http_transport import SimpleHTTPTransport  # noqa: F401
_AIClient = AIClient

# Re-export low-level helper methods so users can import them directly from the
# convenience module.  They are **unbound** functions that expect the
# *AIClient* instance as the first positional argument – i.e. they behave like
# regular methods when called via an instance but can also be imported for
# power-users.
llm_generate_once = _AIClient.llm_generate_once  # type: ignore[attr-defined]
llm_stream_once = _AIClient.llm_stream_once  # type: ignore[attr-defined]
tools_enabled = _AIClient.tools_enabled  # type: ignore[attr-defined]
append_assistant_message = _AIClient.append_assistant_message  # type: ignore[attr-defined]

__all__: list[str] = [
    "AIClient",
    "ToolCallRequest",
    "ToolCallResult",
    "SimpleHTTPTransport",
    "llm_generate_once",
    "llm_stream_once",
    "tools_enabled",
    "append_assistant_message",
] 