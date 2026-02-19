"""Regression tests for OpenAI Responses provider request safety."""


class TestOpenAIResponsesToolFormatting:
    """Ensure tool formatting does not emit invalid strict payloads by default."""

    def test_tools_do_not_default_to_strict(self):
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_events",
                    "description": "Get events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "series_ticker": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["series_ticker"],
                    },
                },
            }
        ]

        formatted = mixin._format_tools_responses(tools)

        assert len(formatted) == 1
        assert "strict" not in formatted[0]
        assert formatted[0]["parameters"]["additionalProperties"] is False

    def test_explicit_strict_true_preserved(self):
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_events",
                    "description": "Get events",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "series_ticker": {"type": "string"},
                        },
                        "required": ["status", "series_ticker"],
                    },
                },
            }
        ]

        formatted = mixin._format_tools_responses(tools)

        assert formatted[0]["strict"] is True

    def test_incompatible_strict_schema_is_auto_downgraded(self):
        from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

        mixin = _OpenAIProviderMixin()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_events",
                    "description": "Get events",
                    "strict": True,
                    "parameters": {"type": "object"},
                },
            }
        ]

        formatted = mixin._format_tools_responses(tools)

        assert formatted[0]["strict"] is False


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
