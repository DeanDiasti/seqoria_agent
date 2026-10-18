"""Direct provider users receive credential-safe errors and diagnostic chains."""

import json
import traceback

import httpx
import pytest

from seqoria_agent._redaction import redact_text
from seqoria_agent.providers import (
    AnthropicProvider,
    AsyncAnthropicProvider,
    AsyncOllamaProvider,
    AsyncOpenAICompatibleProvider,
    AsyncOpenAIProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from seqoria_agent.providers.errors import InvalidRequestError

PROVIDERS = [
    AsyncOpenAIProvider,
    OpenAIProvider,
    AsyncOpenAICompatibleProvider,
    OpenAICompatibleProvider,
    AsyncAnthropicProvider,
    AnthropicProvider,
    AsyncOllamaProvider,
    OllamaProvider,
]
_SECRET = "synthetic-sensitive-value"
_DIAGNOSTIC = (
    f"Rejected upstream https://provider.test/api?api_key={_SECRET}; "
    f"Authorization: Bearer {_SECRET}"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("secret_request_url", [False, True], ids=["safe-cause", "unsafe-cause"])
async def test_direct_provider_error_messages_and_chains_are_safe(
    provider_class, streaming, secret_request_url, monkeypatch
):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200, json={"data": [{"id": "offline"}], "models": [{"name": "offline"}]}
            )
        return httpx.Response(
            400,
            json={
                "error": {"message": _DIAGNOSTIC, "type": "invalid_request_error"},
                "message": _DIAGNOSTIC,
            },
        )

    original_client = httpx.Client
    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client(*args, **kwargs):
        return original_client(*args, **kwargs, transport=transport)

    def async_client(*args, **kwargs):
        return original_async_client(*args, **kwargs, transport=transport)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    base_url = "https://provider.test"
    if secret_request_url:
        base_url += f"/?api_key={_SECRET}"
    kwargs = {"host" if "Ollama" in provider_class.__name__ else "base_url": base_url}
    provider = provider_class(model_id="offline", api_key="synthetic-key", max_retries=0, **kwargs)
    is_async = provider_class.__name__.startswith("Async")
    try:
        with pytest.raises(InvalidRequestError) as caught:
            if is_async:
                if streaming:
                    _ = [chunk async for chunk in provider.async_stream(chat_history=[])]
                else:
                    await provider.async_generate(chat_history=[])
            elif streaming:
                list(provider.stream(chat_history=[]))
            else:
                provider.generate(chat_history=[])
        error = caught.value
        assert "Rejected upstream" in str(error)
        assert "provider.test/api" in str(error)
        assert _SECRET not in str(error)
        assert _SECRET not in "".join(traceback.format_exception(error))
        if secret_request_url:
            assert error.__cause__ is None
            assert error.__suppress_context__
        else:
            assert isinstance(error.__cause__, httpx.HTTPStatusError)
    finally:
        if is_async:
            await provider.aclose()
        else:
            provider.close()


@pytest.mark.parametrize("key", ["password", "Authorization", "api_key"])
def test_quoted_credentials_containing_escaped_quotes_are_redacted(key):
    text = json.dumps({key: f'public "quoted" {_SECRET}'})
    assert _SECRET not in redact_text(text)
