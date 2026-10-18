"""Own a bounded model/tool loop and its history, entirely offline."""

import asyncio

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models.chunk import AssistantMessage, ToolCallRequestList, ToolCallResult
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, ToolCallList

toolbox = LocalToolbox()


@toolbox.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class ScriptedProvider(AsyncBaseProvider):
    """Request a tool, then read its result; this is not a real language model."""

    async def async_generate(self, chat_history, **kwargs):
        last_message = chat_history[-1]
        if isinstance(last_message, ToolCallResult):
            return LLMResponse(
                model_id="offline", type="llm_response", text=f"2 + 3 = {last_message.result}"
            )
        return ToolCallList(
            model_id="offline",
            tool_calls=[
                ToolCall(
                    id="add-1",
                    type="function_call",
                    function_name="add",
                    arguments={"a": 2, "b": 3},
                )
            ],
            usage=None,
        )

    async def async_stream(self, chat_history, **kwargs):
        yield await self.async_generate(chat_history=chat_history, **kwargs)


async def main() -> None:
    history = [{"role": "user", "content": "What is 2 + 3?"}]
    async with (
        ScriptedProvider() as provider,
        AsyncAIClient(providers=[provider], local_toolbox=toolbox) as client,
    ):
        for _ in range(3):
            step = await client.llm_generate_once(history=history)
            if isinstance(step, AssistantMessage):
                history.append(step)
                print("Assistant:", step.content)
                return
            if isinstance(step, ToolCallRequestList):
                outcomes = await client.run_tool_calls(step.tool_calls)
                for request, outcome in zip(step.tool_calls, outcomes, strict=True):
                    # Preserve requests and outcomes, including ToolCallError, for the next model step.
                    history.extend([request, outcome])
                    print(
                        f"Tool: {request.name} -> {outcome.result if isinstance(outcome, ToolCallResult) else outcome.error}"
                    )
        raise RuntimeError("The model did not finish within the turn budget")


if __name__ == "__main__":
    asyncio.run(main())
