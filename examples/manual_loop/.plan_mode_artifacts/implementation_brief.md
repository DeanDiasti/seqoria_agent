## Implementation Brief: Improving Agent Memory Orchestration

### Requirements
- Show active plan injection
- Show tool cycle
- Show plan status changes after each assistant step

### Technical Design
- Rely on memory-level auto planning via `begin_turn`, `record_tool_cycle`, and `commit_assistant_response`
- Print active plan blocks from `messages_for_llm()`
- Execute tool loop for concrete evidence

This design ensures that plan orchestration, tool cycles, and status changes are transparently visible and traceable during each assistant step.
