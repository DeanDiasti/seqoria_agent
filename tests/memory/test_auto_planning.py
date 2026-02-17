from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig, PlanConfig
from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest, ToolCallResult


def test_complex_prompt_auto_creates_plan():
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=True,
                complexity_min_words=10,
                max_inferred_steps=4,
            )
        )
    )

    messages = memory.begin_turn(
        "Compare Q3 and Q4 performance, identify risks, and provide a three-step execution approach."
    )

    assert memory.current_plan is not None
    assert memory.current_plan.source == "inferred"
    assert len(memory.current_plan.steps) >= 2
    system_blocks = [message for message in messages if message.get("role") == "system"]
    assert system_blocks
    assert "<active_plan>" in system_blocks[0]["content"]


def test_simple_prompt_does_not_force_plan():
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=True,
                complexity_min_words=25,
            )
        )
    )

    memory.begin_turn("What is 2 + 2?")
    assert memory.current_plan is None


def test_tool_success_auto_progresses_plan_step():
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=False,
                update_on_tool_success=True,
            )
        )
    )
    memory.set_plan(
        "Build demo output",
        ["Collect requirements", "Draft design"],
        step_metadata=[{"required_tools": ["collect_requirements"]}, {}],
    )

    call = ToolCallRequest(id="tc-1", name="collect_requirements", arguments={"topic": "demo"})
    result = ToolCallResult(tool_call_id="tc-1", name="collect_requirements", result="Requirements captured.")
    memory.record_tool_cycle([call], [result])

    assert memory.current_plan is not None
    step = memory.current_plan.steps[0]
    assert step.status == "completed"
    assert "collect_requirements" in (step.result_summary or "")
    assert step.evidence


def test_assistant_only_progress_can_complete_step():
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=False,
                assistant_progress_enabled=True,
            )
        )
    )
    memory.set_plan("Deliver summary", ["Draft concise summary", "Review and finalize"])
    memory.commit_assistant_response("Completed the draft summary and delivered a concise output.")

    assert memory.current_plan is not None
    assert memory.current_plan.steps[0].status == "completed"


def test_failed_tool_marks_blocked_and_appends_replan_step():
    memory = AgenticMemory(
        config=AgenticMemoryConfig(
            plan=PlanConfig(
                enabled=True,
                auto_infer=False,
                update_on_tool_success=True,
                replan_on_blocked=True,
                max_inferred_steps=5,
            )
        )
    )
    memory.set_plan(
        "Download and process report",
        ["Download source report", "Process report data"],
        step_metadata=[{"required_tools": ["download_report"]}, {}],
    )

    call = ToolCallRequest(id="tc-err", name="download_report", arguments={"report_id": "abc"})
    error = ToolCallError(tool_call_id="tc-err", name="download_report", error="Timeout while fetching report")
    memory.record_tool_cycle([call], [error])

    assert memory.current_plan is not None
    assert memory.current_plan.steps[0].status == "blocked"
    assert len(memory.current_plan.steps) == 3
    assert "Resolve blocker" in memory.current_plan.steps[-1].description
    assert memory.current_plan.steps[-1].metadata.get("source") == "auto_replan"

