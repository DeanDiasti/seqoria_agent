from __future__ import annotations

"""Demonstrate automatic system-prompt insertion and streaming tool results.

Run:

```bash
python examples/manual_loop/system_prompt_demo.py
```
"""

import asyncio
from typing import List

from seqoria_agent.client import AIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.llm_models import ToolCallRequest
from seqoria_agent.llm_providers.base_provider import BaseProvider
from seqoria_agent.llm_providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    async def async_generate(self, *, chat_history, tools=None, tool_choice="auto", model=None):  # noqa: ANN001
        return LLMResponse(model_id="dummy", type="llm_response", text="Hello! (dummy)")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "Hello! (dummy)"
            yield LLMResponse(model_id="dummy", type="llm_response", text="Hello! (dummy)")
        return _gen()


tb = LocalToolbox()


@tb.tool()
async def echo(message: str) -> str:
    return message


async def main() -> None:  # noqa: D401
    custom_prompt = "You are a very friendly assistant that always says please and thank you."

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb, system_prompt=custom_prompt) as ai:
        history: List[dict] = []

        # First illustrate that the system prompt was automatically injected
        await ai.llm_generate_once(history=history, use_tools=False)
        print("System prompt in history →", history[0])

        # Show streaming execution unchanged
        tool_calls = [ToolCallRequest(id="1", name="echo", arguments={"message": "Hi"})]
        async for req, res in ai.executor.run_call_stream(tool_calls=tool_calls):
            print(f"Tool {req.name} returned: {res.result}")


if __name__ == "__main__":
    asyncio.run(main()) 