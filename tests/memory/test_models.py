from seqoria_agent.memory.models import MemoryScope, Plan, PlanStep


def test_memory_scope_key_and_match():
    left = MemoryScope(
        tenant_id="tenant-a",
        user_id="user-1",
        conversation_id="conv-x",
        session_id="sess-1",
    )
    right = MemoryScope(
        tenant_id="tenant-a",
        user_id="user-1",
        conversation_id="conv-x",
        session_id="sess-1",
    )
    other = MemoryScope(tenant_id="tenant-b", user_id="user-1")

    assert left.key() == right.key()
    assert left.matches(right) is True
    assert left.matches(other) is False


def test_plan_progress_summary_and_current_step():
    plan = Plan(
        id="p1",
        goal="Ship feature",
        steps=[
            PlanStep(id="s1", description="Design", status="completed"),
            PlanStep(id="s2", description="Implement", status="in_progress"),
            PlanStep(id="s3", description="Test", status="pending"),
        ],
        current_step_index=1,
    )

    assert plan.current_step is not None
    assert plan.current_step.description == "Implement"
    assert "2 of 3" in plan.progress_summary
    assert "1 completed" in plan.progress_summary

