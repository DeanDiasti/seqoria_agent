"""Ollama model discovery and pull use the same HTTP paths in both variants."""

import json

import httpx
import pytest

from seqoria_agent.providers.errors import ModelNotFoundError
from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin


CASES = [
    ("gemma3", True, False),
    ("gemma3:latest", True, False),
    ("new-model", True, False),
    ("new-model", True, True),
    ("new-model", False, False),
]


def mock_ollama(*, fail_pull):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/api/tags":
            assert request.method == "GET"
            return httpx.Response(200, json={"models": [{"name": "gemma3:latest"}]})
        assert request.method == "POST" and request.url.path == "/api/pull"
        assert json.loads(request.content) == {"model": "new-model", "stream": True}
        events = [{"error": "model not found"}] if fail_pull else [{"status": "pulling"}, {"status": "success"}]
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    return httpx.MockTransport(handler), calls


def assert_requests(calls, *, model, auto_pull):
    assert len(calls) == (2 if model == "new-model" and auto_pull else 1)


@pytest.mark.parametrize("model,auto_pull,fail_pull", CASES)
def test_ollama_model_validation_sync(model, auto_pull, fail_pull):
    transport, calls = mock_ollama(fail_pull=fail_pull)
    with httpx.Client(base_url="http://ollama.test", transport=transport) as client:
        mixin = _OllamaProviderMixin()
        if fail_pull or not auto_pull:
            with pytest.raises(ModelNotFoundError) as exc:
                mixin._validate_model_sync(client, model, auto_pull)
            assert exc.value.available_models == ["gemma3:latest"]
        else:
            mixin._validate_model_sync(client, model, auto_pull)
    assert_requests(calls, model=model, auto_pull=auto_pull)


@pytest.mark.asyncio
@pytest.mark.parametrize("model,auto_pull,fail_pull", CASES)
async def test_ollama_model_validation_async(model, auto_pull, fail_pull):
    transport, calls = mock_ollama(fail_pull=fail_pull)
    async with httpx.AsyncClient(base_url="http://ollama.test", transport=transport) as client:
        mixin = _OllamaProviderMixin()
        if fail_pull or not auto_pull:
            with pytest.raises(ModelNotFoundError) as exc:
                await mixin._validate_model_async(client, model, auto_pull)
            assert exc.value.available_models == ["gemma3:latest"]
        else:
            await mixin._validate_model_async(client, model, auto_pull)
    assert_requests(calls, model=model, auto_pull=auto_pull)
