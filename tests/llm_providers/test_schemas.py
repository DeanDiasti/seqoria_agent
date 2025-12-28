from seqoria_agent.llm_providers.schemas import (
    ToolCall,
    LLMUsage,
    ToolCallList,
    LLMResponse,
)


def test_toolcall_and_lists():
    tc = ToolCall(
        id="abc", type="function", function_name="do_something", arguments={"x": 1}
    )
    usage = LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    tcl = ToolCallList(model_id="model", tool_calls=[tc], usage=usage)

    assert tcl.model_id == "model"
    assert tcl.tool_calls[0].function_name == "do_something"
    assert tcl.usage.total_tokens == 5


def test_llmresponse():
    resp = LLMResponse(
        model_id="m", type="llm_response", text="hello", usage=None, stop_reason="end"
    )
    assert resp.text == "hello"
    assert resp.stop_reason == "end" 