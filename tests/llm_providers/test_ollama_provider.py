"""Comprehensive tests for Ollama provider implementation."""

import pytest
import base64
from unittest.mock import patch

from seqoria_agent.providers.errors import (
    InvalidRequestError,
    ModelNotFoundError,
    ConnectionError,
    LLMError,
)


class TestOllamaProviderInit:
    """Tests for Ollama provider initialization."""

    def test_model_id_required(self):
        """OllamaProvider should raise InvalidRequestError when model_id is not provided."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id=None, host=None, api_key=None)
            
            assert "model_id is required" in str(exc_info.value)

    def test_model_id_required_empty_string(self):
        """OllamaProvider should raise InvalidRequestError when model_id is empty string."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id="", host=None, api_key=None)
            
            assert "model_id is required" in str(exc_info.value)

    def test_default_host(self):
        """Should use default host when not provided."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            host, headers, model_id, default_params, max_retries, auto_pull = mixin._init_common(
                model_id="gemma3", host=None, api_key=None
            )
            
            assert host == "http://localhost:11434"
            assert headers == {}
            assert model_id == "gemma3"

    def test_env_host(self):
        """Should use OLLAMA_HOST from environment."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:8080"}):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            host, _, _, _, _, _ = mixin._init_common(model_id="gemma3", host=None, api_key=None)
            
            assert host == "http://custom:8080"

    def test_explicit_host_overrides_env(self):
        """Explicit host should override environment variable."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://env:8080"}):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            host, _, _, _, _, _ = mixin._init_common(
                model_id="gemma3", host="http://explicit:9090", api_key=None
            )
            
            assert host == "http://explicit:9090"

    def test_api_key_creates_auth_header(self):
        """API key should create Authorization header."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, headers, _, _, _, _ = mixin._init_common(
                model_id="gemma3", host=None, api_key="test-api-key"
            )
            
            assert headers["Authorization"] == "Bearer test-api-key"

    def test_env_api_key(self):
        """Should use OLLAMA_API_KEY from environment."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "env-api-key"}):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, headers, _, _, _, _ = mixin._init_common(
                model_id="gemma3", host=None, api_key=None
            )
            
            assert headers["Authorization"] == "Bearer env-api-key"

    def test_no_default_params(self):
        """Ollama provider should have empty default_params when no kwargs provided."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, _, _, default_params, _, _ = mixin._init_common(
                model_id="gemma3", host=None, api_key=None
            )
            
            assert default_params == {}

    def test_valid_params_stored(self):
        """Valid generation params should be stored in default_params."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, _, _, default_params, _, _ = mixin._init_common(
                model_id="gemma3",
                host=None,
                api_key=None,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                num_predict=1024
            )
            
            assert default_params["temperature"] == 0.7
            assert default_params["top_p"] == 0.9
            assert default_params["top_k"] == 40
            assert default_params["num_predict"] == 1024

    def test_invalid_params_ignored(self):
        """Invalid params should not be stored."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, _, _, default_params, _, _ = mixin._init_common(
                model_id="gemma3",
                host=None,
                api_key=None,
                temperature=0.7,
                invalid_param="ignored",
                max_tokens=1024  # OpenAI param, not Ollama
            )
            
            assert "temperature" in default_params
            assert "invalid_param" not in default_params
            assert "max_tokens" not in default_params

    def test_auto_pull_default_true(self):
        """auto_pull should default to True."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, _, _, _, _, auto_pull = mixin._init_common(
                model_id="gemma3", host=None, api_key=None
            )
            
            assert auto_pull is True

    def test_auto_pull_can_be_disabled(self):
        """auto_pull can be set to False."""
        with patch.dict("os.environ", {}, clear=True):
            from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
            
            mixin = _OllamaProviderMixin()
            _, _, _, _, _, auto_pull = mixin._init_common(
                model_id="gemma3", host=None, api_key=None, auto_pull=False
            )
            
            assert auto_pull is False


class TestOllamaImageConversion:
    """Tests for Ollama image content conversion."""

    def test_convert_string_content(self):
        """String content should return as-is with empty images."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        text, images = mixin._convert_image_content_ollama("Hello, world!")
        
        assert text == "Hello, world!"
        assert images == []

    def test_convert_text_block(self):
        """Text content blocks should be extracted."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        content = [{"type": "text", "text": "What is this?"}]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert text == "What is this?"
        assert images == []

    def test_convert_image_base64(self):
        """Base64 image should be extracted to images list."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        content = [{"type": "image", "base64": test_base64, "media_type": "image/png"}]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert text == ""
        assert len(images) == 1
        assert images[0] == test_base64

    def test_convert_image_data_url(self):
        """Data URL should be parsed and base64 extracted."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        data_url = f"data:image/png;base64,{test_base64}"
        content = [{"type": "image", "url": data_url}]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert len(images) == 1
        assert images[0] == test_base64

    def test_convert_image_regular_url(self):
        """Regular URL should be noted in text (Ollama requires base64)."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        content = [{"type": "image", "url": "https://example.com/image.jpg"}]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert "[Image URL:" in text
        assert images == []

    def test_convert_openai_format_data_url(self):
        """OpenAI image_url format with data URL should be converted."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        content = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{test_base64}"}
        }]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert len(images) == 1
        assert images[0] == test_base64

    def test_convert_mixed_content(self):
        """Mixed text and image content should be converted correctly."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        content = [
            {"type": "text", "text": "Look at this image:"},
            {"type": "image", "base64": test_base64, "media_type": "image/png"},
            {"type": "text", "text": "What do you see?"}
        ]
        
        text, images = mixin._convert_image_content_ollama(content)
        
        assert "Look at this image:" in text
        assert "What do you see?" in text
        assert len(images) == 1
        assert images[0] == test_base64


class TestOllamaChatHistoryPreparation:
    """Tests for Ollama chat history preparation."""

    def test_prepare_simple_messages(self):
        """Simple text messages should be prepared correctly."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = mixin._prepare_chat_history_sync(history)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_prepare_system_message(self):
        """System messages should be prepared correctly."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = mixin._prepare_chat_history_sync(history)
        
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_prepare_message_with_images(self):
        """Messages with images should include images field."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        
        mixin = _OllamaProviderMixin()
        test_base64 = base64.b64encode(b"fake_image").decode()
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image", "base64": test_base64}
            ]
        }]
        
        result = mixin._prepare_chat_history_sync(history)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "What's this?" in result[0]["content"]
        assert "images" in result[0]
        assert len(result[0]["images"]) == 1

    def test_prepare_tool_call_request(self):
        """ToolCallRequest models should be converted to assistant messages."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        from seqoria_agent.models.chunk import ToolCallRequest
        
        mixin = _OllamaProviderMixin()
        history = [
            ToolCallRequest(id="1", name="get_weather", arguments={"city": "NYC"})
        ]
        
        result = mixin._prepare_chat_history_sync(history)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_prepare_tool_result(self):
        """ToolCallResult models should be converted to tool messages."""
        from seqoria_agent.providers.ollama_provider import _OllamaProviderMixin
        from seqoria_agent.models.chunk import ToolCallResult
        
        mixin = _OllamaProviderMixin()
        history = [
            ToolCallResult(tool_call_id="1", result="Sunny, 72°F")
        ]
        
        result = mixin._prepare_chat_history_sync(history)
        
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Sunny, 72°F"


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
        tools = [{
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }]
        
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
        tool_calls = [{
            "function": {
                "name": "get_weather",
                "arguments": {"city": "NYC"}
            }
        }]
        
        result = mixin._parse_tool_calls(tool_calls)
        
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].function_name == "get_weather"
        assert result[0].arguments == {"city": "NYC"}


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
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "format": "json",
            "keep_alive": "5m"
        }
        
        top_level, options = mixin._build_options(params)
        
        assert top_level["format"] == "json"
        assert top_level["keep_alive"] == "5m"
        assert options["temperature"] == 0.7
        assert options["top_p"] == 0.9

