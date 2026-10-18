"""Model-specific request parameters and Ollama generation options."""


class TestOpenAIResponsesParamSanitization:
    """Ensure model-incompatible params are removed before request send."""

    def test_gpt5_models_drop_temperature(self):
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        params = {"temperature": 0.4, "top_p": 0.9}

        sanitized = mixin._sanitize_params_for_model("gpt-5-mini-2025-08-07", params)

        assert "temperature" not in sanitized
        assert sanitized["top_p"] == 0.9
        assert params["temperature"] == 0.4

    def test_non_gpt5_models_keep_temperature(self):
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        params = {"temperature": 0.4, "top_p": 0.9}

        sanitized = mixin._sanitize_params_for_model("gpt-4o", params)

        assert sanitized["temperature"] == 0.4
        assert sanitized["top_p"] == 0.9


class TestOllamaOptionsBuilding:
    """Tests for building Ollama options from params."""

    def test_build_options_empty(self):
        """Empty params should return empty dicts."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        top_level, options = mixin._build_options({})

        assert top_level == {}
        assert options == {}

    def test_build_options_separates_correctly(self):
        """Params should be separated into top-level and options."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        params = {"temperature": 0.7, "top_p": 0.9, "format": "json", "keep_alive": "5m"}

        top_level, options = mixin._build_options(params)

        assert top_level["format"] == "json"
        assert top_level["keep_alive"] == "5m"
        assert options["temperature"] == 0.7
        assert options["top_p"] == 0.9
