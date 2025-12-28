"""Tests for model validation functionality in providers."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from seqoria_agent.providers.errors import ModelNotFoundError


class TestOpenAIModelValidation:
    """Tests for OpenAI model validation."""

    def test_validate_model_sync_success(self):
        """Model validation should pass when model exists."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        mixin.model_id = "gpt-4o"
        
        # Mock client with models list
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        # Should not raise
        mixin._validate_model_sync(client=mock_client, model_id="gpt-4o")

    def test_validate_model_sync_not_found(self):
        """Model validation should raise ModelNotFoundError when model doesn't exist."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        mixin.model_id = "nonexistent-model"
        
        # Mock client with models list
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            mixin._validate_model_sync(client=mock_client, model_id="nonexistent-model")
        
        error = exc_info.value
        assert error.model_id == "nonexistent-model"
        assert error.provider == "OpenAI"
        assert "gpt-4o" in error.available_models
        assert "gpt-3.5-turbo" in error.available_models

    def test_validate_model_sync_error_message_includes_available_models(self):
        """Error message should list all available models."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        mixin.model_id = "nonexistent-model"
        
        # Mock client with models list
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            mixin._validate_model_sync(client=mock_client, model_id="nonexistent-model")
        
        error_message = str(exc_info.value)
        assert "Available models:" in error_message
        assert "gpt-4o" in error_message
        assert "gpt-3.5-turbo" in error_message

    @pytest.mark.asyncio
    async def test_validate_model_async_success(self):
        """Async model validation should pass when model exists."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        mixin.model_id = "gpt-4o"
        
        # Mock async client with models list
        mock_client = AsyncMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        # Should not raise
        await mixin._validate_model_async(async_client=mock_client, model_id="gpt-4o")

    @pytest.mark.asyncio
    async def test_validate_model_async_not_found(self):
        """Async model validation should raise ModelNotFoundError when model doesn't exist."""
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin
        
        mixin = _OpenAIProviderMixin()
        mixin.model_id = "nonexistent-model"
        
        # Mock async client with models list
        mock_client = AsyncMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            await mixin._validate_model_async(async_client=mock_client, model_id="nonexistent-model")
        
        error = exc_info.value
        assert error.model_id == "nonexistent-model"
        assert error.provider == "OpenAI"


class TestAnthropicModelValidation:
    """Tests for Anthropic model validation."""

    def test_validate_model_sync_success(self):
        """Model validation should pass when model exists."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        mixin.model_id = "claude-sonnet-4-20250514"
        
        # Mock client with models list
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "claude-sonnet-4-20250514"
        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-haiku-20240307"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        # Should not raise
        mixin._validate_model_sync(client=mock_client, model_id="claude-sonnet-4-20250514")

    def test_validate_model_sync_not_found(self):
        """Model validation should raise ModelNotFoundError when model doesn't exist."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        mixin.model_id = "nonexistent-model"
        
        # Mock client with models list
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "claude-sonnet-4-20250514"
        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-haiku-20240307"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            mixin._validate_model_sync(client=mock_client, model_id="nonexistent-model")
        
        error = exc_info.value
        assert error.model_id == "nonexistent-model"
        assert error.provider == "Anthropic"
        assert "claude-sonnet-4-20250514" in error.available_models
        assert "claude-3-haiku-20240307" in error.available_models

    @pytest.mark.asyncio
    async def test_validate_model_async_success(self):
        """Async model validation should pass when model exists."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        mixin.model_id = "claude-sonnet-4-20250514"
        
        # Mock async client with models list
        mock_client = AsyncMock()
        mock_model1 = MagicMock()
        mock_model1.id = "claude-sonnet-4-20250514"
        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-haiku-20240307"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        # Should not raise
        await mixin._validate_model_async(async_client=mock_client, model_id="claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    async def test_validate_model_async_not_found(self):
        """Async model validation should raise ModelNotFoundError when model doesn't exist."""
        from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin
        
        mixin = _AnthropicProviderMixin()
        mixin.model_id = "nonexistent-model"
        
        # Mock async client with models list
        mock_client = AsyncMock()
        mock_model1 = MagicMock()
        mock_model1.id = "claude-sonnet-4-20250514"
        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-haiku-20240307"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        
        with pytest.raises(ModelNotFoundError) as exc_info:
            await mixin._validate_model_async(async_client=mock_client, model_id="nonexistent-model")
        
        error = exc_info.value
        assert error.model_id == "nonexistent-model"
        assert error.provider == "Anthropic"


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

