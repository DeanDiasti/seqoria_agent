"""Exercise model validation through the providers' actual HTTP interface."""

import json

import httpx
import pytest

from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
from seqoria_agent.providers.errors import ModelNotFoundError, QuotaExceededError
from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

HTTP_MODEL_CASES = [
    (_OpenAIProviderMixin, "OpenAI", "/models"),
    (_AnthropicProviderMixin, "Anthropic", "/v1/models"),
]


def model_listing(request, *, endpoint, status):
    assert request.method == "GET"
    assert request.url.path == endpoint
    return httpx.Response(status, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})


def assert_missing(error, provider):
    assert error.model_id == "missing"
    assert error.provider == provider
    assert error.available_models == ["model-a", "model-b"]
    assert "Available models:" in str(error)
    assert "model-a" in str(error) and "model-b" in str(error)


@pytest.mark.parametrize("mixin_type,provider,endpoint", HTTP_MODEL_CASES)
@pytest.mark.parametrize("model,status", [("model-a", 200), ("missing", 200), ("model-a", 429)])
def test_validate_model_http_sync(mixin_type, provider, endpoint, model, status):
    transport = httpx.MockTransport(
        lambda request: model_listing(request, endpoint=endpoint, status=status)
    )
    with httpx.Client(base_url="https://provider.test", transport=transport) as client:
        mixin = mixin_type()
        if status == 429:
            with pytest.raises(QuotaExceededError):
                mixin._validate_model_sync(client, model)
        elif model == "missing":
            with pytest.raises(ModelNotFoundError) as exc:
                mixin._validate_model_sync(client, model)
            assert_missing(exc.value, provider)
        else:
            mixin._validate_model_sync(client, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("mixin_type,provider,endpoint", HTTP_MODEL_CASES)
@pytest.mark.parametrize("model,status", [("model-a", 200), ("missing", 200), ("model-a", 429)])
async def test_validate_model_http_async(mixin_type, provider, endpoint, model, status):
    transport = httpx.MockTransport(
        lambda request: model_listing(request, endpoint=endpoint, status=status)
    )
    async with httpx.AsyncClient(base_url="https://provider.test", transport=transport) as client:
        mixin = mixin_type()
        if status == 429:
            with pytest.raises(QuotaExceededError):
                await mixin._validate_model_async(client, model)
        elif model == "missing":
            with pytest.raises(ModelNotFoundError) as exc:
                await mixin._validate_model_async(client, model)
            assert_missing(exc.value, provider)
        else:
            await mixin._validate_model_async(client, model)


OLLAMA_MODEL_CASES = [
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
        events = (
            [{"error": "model not found"}]
            if fail_pull
            else [{"status": "pulling"}, {"status": "success"}]
        )
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    return httpx.MockTransport(handler), calls


def assert_requests(calls, *, model, auto_pull):
    assert len(calls) == (2 if model == "new-model" and auto_pull else 1)


@pytest.mark.parametrize("model,auto_pull,fail_pull", OLLAMA_MODEL_CASES)
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
@pytest.mark.parametrize("model,auto_pull,fail_pull", OLLAMA_MODEL_CASES)
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
