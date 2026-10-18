# Provider API

All retained providers use HTTPX and ship in the core installation. No vendor SDK is required. Import classes from `seqoria_agent.providers`; each backend exposes async and synchronous variants. Use async providers with `AsyncAIClient`. `AIClient` adapts either variant for blocking scripts.

| Backend | Async class | Sync class | Service required |
| --- | --- | --- | --- |
| OpenAI | `AsyncOpenAIProvider` | `OpenAIProvider` | OpenAI API credentials and model access |
| OpenAI-compatible HTTP | `AsyncOpenAICompatibleProvider` | `OpenAICompatibleProvider` | Compatible endpoint and its credentials |
| Anthropic | `AsyncAnthropicProvider` | `AnthropicProvider` | Anthropic API credentials and model access |
| Ollama | `AsyncOllamaProvider` | `OllamaProvider` | Running Ollama service and a suitable model |

Use explicit model IDs supported by the endpoint. OpenAI-compatible services differ in tool, streaming, image, and reasoning support; an accepted HTTP schema does not guarantee all capabilities. Default tests use deterministic HTTP transports. They do not certify every hosted model or endpoint.

The application owns provider resources. Enter provider context managers before client context managers, then close clients before providers. A provider must stay within its owning event loop when used asynchronously. Failed streams after partial output raise rather than replaying output through a fallback provider.

## OpenAI

::: seqoria_agent.providers.openai_provider.AsyncOpenAIProvider

::: seqoria_agent.providers.openai_provider.OpenAIProvider

## OpenAI-compatible HTTP

::: seqoria_agent.providers.openai_compatible_provider.AsyncOpenAICompatibleProvider

::: seqoria_agent.providers.openai_compatible_provider.OpenAICompatibleProvider

## Anthropic

::: seqoria_agent.providers.anthropic_provider.AsyncAnthropicProvider

::: seqoria_agent.providers.anthropic_provider.AnthropicProvider

## Ollama

::: seqoria_agent.providers.ollama_provider.AsyncOllamaProvider

::: seqoria_agent.providers.ollama_provider.OllamaProvider

## Custom provider interfaces

::: seqoria_agent.providers.base_provider

## Factory helpers

::: seqoria_agent.providers.get_provider

::: seqoria_agent.providers.list_available_providers

`list_available_providers()` checks local dependency availability, not endpoint health, credentials, installed models, or tool support.

## Schemas and errors

::: seqoria_agent.providers.schemas

::: seqoria_agent.providers.errors
