"""Exercise model validation through the providers' actual HTTP interface."""

import httpx
import pytest

from seqoria_agent.providers.errors import ModelNotFoundError, QuotaExceededError
from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin


CASES = [
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


@pytest.mark.parametrize("mixin_type,provider,endpoint", CASES)
@pytest.mark.parametrize("model,status", [("model-a", 200), ("missing", 200), ("model-a", 429)])
def test_validate_model_http_sync(mixin_type, provider, endpoint, model, status):
    transport = httpx.MockTransport(lambda request: model_listing(request, endpoint=endpoint, status=status))
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
@pytest.mark.parametrize("mixin_type,provider,endpoint", CASES)
@pytest.mark.parametrize("model,status", [("model-a", 200), ("missing", 200), ("model-a", 429)])
async def test_validate_model_http_async(mixin_type, provider, endpoint, model, status):
    transport = httpx.MockTransport(lambda request: model_listing(request, endpoint=endpoint, status=status))
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


class TestModelNotFoundErrorMessage:
    """Tests for ModelNotFoundError message formatting."""

    def test_error_without_available_models(self):
        """Error message should work without available models."""
        error = ModelNotFoundError(model_id="test-model", provider="TestProvider")
        
        assert "test-model" in str(error)
        assert "TestProvider" in str(error)
        assert error.available_models == []

    def test_error_with_available_models(self):
        """Error message should include available models when provided."""
        error = ModelNotFoundError(
            model_id="test-model",
            provider="TestProvider",
            available_models=["model-a", "model-b", "model-c"]
        )
        
        assert "test-model" in str(error)
        assert "TestProvider" in str(error)
        assert "Available models:" in str(error)
        assert "model-a" in str(error)
        assert "model-b" in str(error)
        assert "model-c" in str(error)
        assert error.available_models == ["model-a", "model-b", "model-c"]

    def test_error_with_fallback_model(self):
        """Error message should include fallback model when provided."""
        error = ModelNotFoundError(
            model_id="test-model",
            provider="TestProvider",
            fallback_model="default-model"
        )
        
        assert "test-model" in str(error)
        assert "fallback" in str(error).lower() or "default-model" in str(error)

