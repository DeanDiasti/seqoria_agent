import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig, PlanConfig
from seqoria_agent.models.chunk import AssistantMessage, ToolCallResult
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList
from seqoria_agent.toolbox import LocalToolbox


class _AutoPlanToolStub(AsyncBaseProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        del chat_history, kwargs
        self.calls += 1
        usage = LLMUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        if self.calls == 1 and tools:
            return ToolCallList(
                model_id="auto-plan-stub",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        function_name="collect_requirements",
                        arguments={"topic": "auto planning"},
                    )
                ],
                usage=usage,
            )
        return LLMResponse(
            model_id="auto-plan-stub",
            type="llm_response",
            text="Completed the requested workflow and produced final guidance.",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        yield await self.async_generate(chat_history=chat_history, tools=tools, **kwargs)


@pytest.mark.asyncio
async def test_chat_auto_infers_and_progresses_plan_through_memory_hooks():
    toolbox = LocalToolbox()

    @toolbox.tool()
    def collect_requirements(topic: str) -> str:  # noqa: ANN001
        return f"requirements gathered for: {topic}"

    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=True,
                update_on_tool_success=True,
                assistant_progress_enabled=True,
                complexity_min_words=6,
            )
        )
    )
    provider = _AutoPlanToolStub()
    ai = AsyncAIClient(urls=[], providers=[provider], local_toolbox=toolbox)

    async with ai as client:
        chunks = [chunk async for chunk in client.chat("Gather requirements, execute tool work, and summarize.", history=memory)]

    assert provider.calls == 2
    assert any(isinstance(chunk, list) for chunk in chunks)
    assert isinstance(chunks[-1], AssistantMessage)
    assert memory.current_plan is not None
    assert memory.current_plan.source == "inferred"
    assert any(step.status == "completed" for step in memory.current_plan.steps)
    payload = memory.to_dict()
    roles = [item["role"] for item in payload["context"]["raw_messages"]]
    assert "tool_call" in roles
    assert "tool_result" in roles
    assert any(isinstance(item, ToolCallResult) for item in chunks[0])
