"""Shared validation for provider configuration and untrusted tool payloads."""

from typing import Any

from seqoria_agent.client.ai_client.utils import json_loads

from .errors import LLMError


def validate_max_retries(value: Any) -> int:
    """Reject ambiguous retry budgets before creating an HTTP client."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    return value


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    """Require a complete JSON object rather than silently executing with {}."""
    if isinstance(value, str):
        try:
            value = json_loads(value)
        except (ValueError, TypeError) as exc:
            raise LLMError("Provider returned malformed tool arguments") from exc
    if not isinstance(value, dict):
        raise LLMError("Provider tool arguments must be a JSON object")
    return value
