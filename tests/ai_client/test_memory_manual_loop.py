import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk, ToolCallRequestList
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList
from seqoria_agent.toolbox import LocalToolbox


class _ManualToolStub(AsyncBaseProvider):
    def __init__(self):
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1 and tools:
            usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return ToolCallList(
                model_id="manual-tool-stub",
                tool_calls=[ToolCall(id="tc-1", function_name="multiply", arguments={"a": 2, "b": 3})],
                usage=usage,
            )
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(
            model_id="manual-tool-stub",
            type="llm_response",
            text="The result is 6.",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        yield LLMResponse(
            model_id="manual-tool-stub",
            type="llm_response",
            text="The result is 6.",
            usage=usage,
            stop_reason="end_turn",
        )


class _ManualStreamStub(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(
            model_id="manual-stream-stub",
            type="llm_response",
            text="stream answer",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        yield LLMResponse(model_id="manual-stream-stub", type="llm_delta", text="stream ")
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        yield LLMResponse(
            model_id="manual-stream-stub",
            type="llm_response",
            text="stream answer",
            usage=usage,
            stop_reason="end_turn",
        )


@pytest.mark.asyncio
async def test_manual_loop_with_llm_generate_once_and_memory_hooks():
    box = LocalToolbox()

    @box.tool()
    def multiply(a: int, b: int) -> int:  # noqa: ANN001
        return a * b

    memory = AgenticMemory()
    ai = AsyncAIClient(urls=[], providers=[_ManualToolStub()], local_toolbox=box)

    async with ai as client:
        _ = memory.begin_turn("multiply 2 and 3", retrieval_query="multiply numbers")
        for _ in range(5):
            response = await client.llm_generate_once(history=memory, use_tools=True)
            if isinstance(response, ToolCallRequestList):
                tool_results = await client.run_tool_calls(response.tool_calls)
                memory.record_tool_cycle(response.tool_calls, tool_results)
                continue
            if isinstance(response, AssistantMessage):
                memory.commit_assistant_response(
                    content=response.content,
                    model=response.model,
                    usage=response.usage,
                )
                memory.end_turn()
                break

    roles = [item["role"] for item in memory.to_dict()["context"]["raw_messages"]]
    assert roles == ["user", "tool_call", "tool_result", "assistant"]


@pytest.mark.asyncio
async def test_manual_loop_with_llm_stream_once_and_memory_hooks():
    memory = AgenticMemory()
    ai = AsyncAIClient(urls=[], providers=[_ManualStreamStub()])

    async with ai as client:
        _ = memory.begin_turn("stream this", retrieval_query="stream")
        streamed_parts: list[str] = []
        async for chunk in client.llm_stream_once(history=memory, use_tools=False):
            if isinstance(chunk, ChatChunk):
                streamed_parts.append(chunk.content)
                continue
            if isinstance(chunk, AssistantMessage):
                memory.commit_assistant_response(
                    content=chunk.content or "".join(streamed_parts),
                    model=chunk.model,
                    usage=chunk.usage,
                )
                memory.end_turn()
                break

    roles = [item["role"] for item in memory.to_dict()["context"]["raw_messages"]]
    assert roles == ["user", "assistant"]

