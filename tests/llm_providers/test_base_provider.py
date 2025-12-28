import asyncio

import pytest

from seqoria_agent.llm_providers.base_provider import BaseProvider
from seqoria_agent.llm_providers.errors import ConnectionError, MaxRetriesExceededError, ServerError


class DummyProvider(BaseProvider):
    """Minimal concrete provider for testing internal helpers."""

    def __init__(self):
        super().__init__(model_id="dummy")
        self.max_retries = 2

    # ------------------------------------------------------------------
    # Required abstract methods – we never call them in tests
    # ------------------------------------------------------------------
    async def async_generate(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def async_stream(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Error mapping helper
    # ------------------------------------------------------------------
    def _handle_error(self, error: Exception):  # noqa: D401  # type: ignore[override]
        # Map everything into ConnectionError to trigger retry
        if isinstance(error, ServerError):
            return error
        return ConnectionError(provider="Dummy")


@pytest.mark.asyncio
async def test_execute_with_retry_success_after_retry():
    provider = DummyProvider()

    attempts = 0

    async def flaky_op():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError(provider="Dummy")
        return "ok"

    result = await provider._execute_with_retry(operation=flaky_op)
    assert result == "ok"
    assert attempts == 2  # one failure, one success


@pytest.mark.asyncio
async def test_execute_with_retry_exhausts_retries():
    provider = DummyProvider()

    async def always_fail():
        raise ServerError(provider="Dummy", status_code=500)

    with pytest.raises(MaxRetriesExceededError):
        await provider._execute_with_retry(operation=always_fail) 