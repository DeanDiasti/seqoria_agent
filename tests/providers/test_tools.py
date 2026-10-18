"""Provider tool schemas, strict mode and tool-call argument conversion."""


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


class TestOllamaToolConversion:
    """Tests for Ollama tool definition conversion."""

    def test_convert_tools_none(self):
        """None tools should return None."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        result = mixin._convert_tools_for_ollama(None)

        assert result is None

    def test_convert_tools_empty(self):
        """Empty tools list should return None."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        result = mixin._convert_tools_for_ollama([])

        assert result is None

    def test_convert_tools_basic(self):
        """Tools should be converted to Ollama's format."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        result = mixin._convert_tools_for_ollama(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather"
        assert "properties" in result[0]["function"]["parameters"]


class TestOllamaToolCallParsing:
    """Tests for parsing tool calls from Ollama responses."""

    def test_parse_tool_calls_none(self):
        """None tool_calls should return empty list."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        result = mixin._parse_tool_calls(None)

        assert result == []

    def test_parse_tool_calls_empty(self):
        """Empty tool_calls should return empty list."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin

        mixin = _OllamaProviderMixin()
        result = mixin._parse_tool_calls([])

        assert result == []

    def test_parse_tool_calls(self):
        """Tool calls should be parsed to ToolCall objects."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        from seqoria_agent.providers.schemas import ToolCall

        mixin = _OllamaProviderMixin()
        tool_calls = [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}]

        result = mixin._parse_tool_calls(tool_calls)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].function_name == "get_weather"
        assert result[0].arguments == {"city": "NYC"}
