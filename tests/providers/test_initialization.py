"""Provider model requirements, environment settings and constructor parameters."""

from unittest.mock import patch

import pytest

from seqoria_agent.providers.errors import InvalidRequestError


class TestOpenAIProviderInit:
    """Tests for OpenAI provider initialization."""

    def test_model_id_required_sync(self):
        """OpenAIProvider should raise InvalidRequestError when model_id is not provided."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id=None, api_key="test-key", base_url=None)

            assert "model_id is required" in str(exc_info.value)

    def test_model_id_required_empty_string(self):
        """OpenAIProvider should raise InvalidRequestError when model_id is empty string."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id="", api_key="test-key", base_url=None)

            assert "model_id is required" in str(exc_info.value)

    def test_no_default_params_openai(self):
        """OpenAI provider should have empty default_params when no kwargs provided."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            init_result = mixin._init_common(model_id="gpt-4o", api_key="test-key", base_url=None)
            default_params = init_result[3]

            # Should be empty - no default values
            assert default_params == {}

    def test_valid_params_stored_openai(self):
        """OpenAI provider should store valid params from kwargs (Responses API)."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            init_result = mixin._init_common(
                model_id="gpt-4o",
                api_key="test-key",
                base_url=None,
                temperature=0.5,
                max_output_tokens=2048,  # Responses API uses max_output_tokens
                top_p=0.9,
            )
            default_params = init_result[3]

            assert default_params["temperature"] == 0.5
            assert default_params["max_output_tokens"] == 2048
            assert default_params["top_p"] == 0.9

    def test_invalid_params_ignored_openai(self):
        """OpenAI provider should ignore invalid params from kwargs."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            init_result = mixin._init_common(
                model_id="gpt-4o",
                api_key="test-key",
                base_url=None,
                temperature=0.5,
                invalid_param="should_be_ignored",
                another_invalid=123,
            )
            default_params = init_result[3]

            assert "temperature" in default_params
            assert "invalid_param" not in default_params
            assert "another_invalid" not in default_params

    def test_all_valid_openai_params(self):
        """All valid OpenAI Responses API params should be accepted."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from seqoria_agent.providers.openai_provider import _OpenAIProviderMixin

            mixin = _OpenAIProviderMixin()
            init_result = mixin._init_common(
                model_id="gpt-4o",
                api_key="test-key",
                base_url=None,
                max_output_tokens=1024,  # Responses API parameter
                temperature=0.7,
                top_p=1.0,
                top_logprobs=5,
                reasoning={"effort": "high"},  # Responses API reasoning config
                truncation="auto",
                store=True,
                service_tier="default",
            )
            default_params = init_result[3]

            assert default_params["max_output_tokens"] == 1024
            assert default_params["temperature"] == 0.7
            assert default_params["top_p"] == 1.0
            assert default_params["top_logprobs"] == 5
            assert default_params["reasoning"] == {"effort": "high"}
            assert default_params["truncation"] == "auto"
            assert default_params["store"] is True
            assert default_params["service_tier"] == "default"


class TestAnthropicProviderInit:
    """Tests for Anthropic provider initialization."""

    def test_model_id_required_sync(self):
        """AnthropicProvider should raise InvalidRequestError when model_id is not provided."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id=None, api_key="test-key")

            assert "model_id is required" in str(exc_info.value)

    def test_model_id_required_empty_string(self):
        """AnthropicProvider should raise InvalidRequestError when model_id is empty string."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            with pytest.raises(InvalidRequestError) as exc_info:
                mixin._init_common(model_id="", api_key="test-key")

            assert "model_id is required" in str(exc_info.value)

    def test_no_default_params_anthropic(self):
        """Anthropic provider should have empty default_params when no kwargs provided."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            _, _, _, default_params, _ = mixin._init_common(
                model_id="claude-sonnet-4-20250514", api_key="test-key"
            )

            # Should be empty - no default values
            assert default_params == {}

    def test_valid_params_stored_anthropic(self):
        """Anthropic provider should store valid params from kwargs."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            _, _, _, default_params, _ = mixin._init_common(
                model_id="claude-sonnet-4-20250514",
                api_key="test-key",
                temperature=0.5,
                max_tokens=2048,
                top_p=0.9,
                top_k=40,
            )

            assert default_params["temperature"] == 0.5
            assert default_params["max_tokens"] == 2048
            assert default_params["top_p"] == 0.9
            assert default_params["top_k"] == 40

    def test_invalid_params_ignored_anthropic(self):
        """Anthropic provider should ignore invalid params from kwargs."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            _, _, _, default_params, _ = mixin._init_common(
                model_id="claude-sonnet-4-20250514",
                api_key="test-key",
                temperature=0.5,
                invalid_param="should_be_ignored",
                frequency_penalty=0.0,  # This is OpenAI-specific, not Anthropic
            )

            assert "temperature" in default_params
            assert "invalid_param" not in default_params
            assert "frequency_penalty" not in default_params

    def test_all_valid_anthropic_params(self):
        """All valid Anthropic params should be accepted."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from seqoria_agent.providers.anthropic_provider import _AnthropicProviderMixin

            mixin = _AnthropicProviderMixin()
            _, _, _, default_params, _ = mixin._init_common(
                model_id="claude-sonnet-4-20250514",
                api_key="test-key",
                max_tokens=1024,
                temperature=0.7,
                top_p=1.0,
                top_k=40,
                stop_sequences=["END"],
                metadata={"user_id": "123"},
            )

            assert default_params["max_tokens"] == 1024
            assert default_params["temperature"] == 0.7
            assert default_params["top_p"] == 1.0
            assert default_params["top_k"] == 40
            assert default_params["stop_sequences"] == ["END"]
            assert default_params["metadata"] == {"user_id": "123"}


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
            _, headers, _, _, _, _ = mixin._init_common(model_id="gemma3", host=None, api_key=None)

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
                num_predict=1024,
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
                max_tokens=1024,  # OpenAI param, not Ollama
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
