"""Provider error taxonomy and informative model-not-found diagnostics."""

from seqoria_agent.providers.errors import ConnectionError, LLMError, ModelNotFoundError


class TestOllamaErrorHandling:
    """Tests for Ollama error handling."""

    def test_handle_connection_error(self):
        """Connection errors should map to ConnectionError."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        mixin.model_id = "test"

        error = Exception("Connection refused")
        result = mixin._handle_error(error)

        assert isinstance(result, ConnectionError)
        assert "Connection" in str(result)

    def test_handle_timeout_error(self):
        """Timeout errors should map to ConnectionError."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        mixin.model_id = "test"

        error = Exception("Request timed out")
        result = mixin._handle_error(error)

        assert isinstance(result, ConnectionError)
        assert "timed out" in str(result).lower()

    def test_handle_model_not_found_error(self):
        """Model not found errors should map to ModelNotFoundError."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        mixin.model_id = "nonexistent"

        error = Exception("Model not found")
        result = mixin._handle_error(error)

        assert isinstance(result, ModelNotFoundError)

    def test_handle_unknown_error(self):
        """Unknown errors should map to generic LLMError."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        mixin.model_id = "test"

        error = Exception("Some weird error")
        result = mixin._handle_error(error)

        assert isinstance(result, LLMError)


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
            available_models=["model-a", "model-b", "model-c"],
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
            model_id="test-model", provider="TestProvider", fallback_model="default-model"
        )

        assert "test-model" in str(error)
        assert "fallback" in str(error).lower() or "default-model" in str(error)
