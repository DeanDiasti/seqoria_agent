"""Stream one response from an explicitly configured external model provider."""

import argparse
import asyncio
import os
from contextlib import aclosing

from seqoria_agent.providers import (
    AsyncAnthropicProvider,
    AsyncOllamaProvider,
    AsyncOpenAICompatibleProvider,
    AsyncOpenAIProvider,
)
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList


def configured_provider(parser: argparse.ArgumentParser, name: str) -> AsyncBaseProvider:
    """Validate environment configuration before creating an HTTP client."""
    model_id = os.environ.get("SEQORIA_MODEL")
    if not model_id:
        parser.error("set SEQORIA_MODEL to a model available on the selected endpoint")
    if name == "ollama":
        # Pull the model yourself; running an example never starts a download.
        return AsyncOllamaProvider(model_id=model_id, auto_pull=False)
    key_variable = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "compatible": "SEQORIA_API_KEY",
    }[name]
    api_key = os.environ.get(key_variable)
    if not api_key:
        parser.error(f"set {key_variable} before running this example")
    if name == "compatible":
        base_url = os.environ.get("SEQORIA_BASE_URL")
        if not base_url:
            parser.error("set SEQORIA_BASE_URL to your Chat Completions endpoint's base URL")
        return AsyncOpenAICompatibleProvider(model_id=model_id, api_key=api_key, base_url=base_url)
    provider_class = AsyncOpenAIProvider if name == "openai" else AsyncAnthropicProvider
    return provider_class(model_id=model_id, api_key=api_key)


async def main(provider: AsyncBaseProvider, prompt: str) -> None:
    printed_delta = False
    async with (
        provider,
        aclosing(provider.async_stream([{"role": "user", "content": prompt}])) as stream,
    ):
        async for chunk in stream:
            if isinstance(chunk, ToolCallList):
                # Provider calls describe tools; an AIClient or application executes them.
                print("Tool requests:", [call.model_dump() for call in chunk.tool_calls])
            elif isinstance(chunk, LLMResponse) and chunk.type == "llm_delta":
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    printed_delta = True
            elif isinstance(chunk, LLMResponse):
                if not printed_delta and chunk.text:
                    print(chunk.text, end="")
                print()
                if chunk.usage:
                    print("Usage:", chunk.usage.model_dump())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", required=True, choices=["openai", "anthropic", "compatible", "ollama"]
    )
    parser.add_argument("--prompt", default="Explain what an agent tool is in one sentence.")
    args = parser.parse_args()
    asyncio.run(main(configured_provider(parser, args.provider), args.prompt))
