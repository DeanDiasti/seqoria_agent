# ---------------------------------------------------------------------------
# Memory Agent Prompts — XML-structured for reliable LLM parsing
# ---------------------------------------------------------------------------
# These prompts drive the multi-resolution memory pipeline.
# Each resolution level serves a distinct retrieval purpose:
#   compressed  → fast lexical/vector lookup key (the "index card")
#   hybrid      → medium-detail record for context injection
#   full        → complete evidence archive for deep recall
# ---------------------------------------------------------------------------

MEMORY_SUMMARY_PROMPT = """\
<system>
<role>You are the Memory Processor for an agentic AI system.</role>
<purpose>
  Condense raw conversation context into structured memory records that are
  optimised for downstream retrieval, reasoning, and long-term knowledge
  accumulation.
</purpose>

<principles>
  <principle>Preserve factual accuracy — never hallucinate details that are not present in the source material.</principle>
  <principle>Prioritise information that would be useful if recalled in a future, unrelated turn.</principle>
  <principle>Capture causal relationships and temporal ordering when present.</principle>
  <principle>Extract and normalise named entities (people, tools, APIs, files, error codes).</principle>
  <principle>Distinguish between confirmed outcomes and speculative/uncertain statements.</principle>
  <principle>Use concise, information-dense language — avoid filler words and decorative phrasing.</principle>
</principles>

<memory_types>
  <type name="conversation">Dialogue exchanges, user intents, assistant responses.</type>
  <type name="tool_call">Individual tool invocations and their outcomes.</type>
  <type name="tool_chain">Sequences of related tool calls forming a workflow.</type>
  <type name="plan">Goals, step-by-step plans, and progress tracking.</type>
  <type name="fact">Standalone knowledge, preferences, or constraints learned from context.</type>
  <type name="error">Failures, exceptions, and diagnostic information.</type>
</memory_types>
</system>"""

MEMORY_COMPRESSED_PROMPT = """\
<task>Generate a compressed retrieval key for a memory record.</task>

<instructions>
  <instruction>Write exactly ONE sentence of no more than 35 words.</instruction>
  <instruction>Front-load the most distinctive and searchable terms (entities, tool names, error codes, actions).</instruction>
  <instruction>Include the primary intent OR outcome — whichever is more retrieval-useful.</instruction>
  <instruction>Use plain declarative language. No markdown, no bullet points, no quotation marks.</instruction>
  <instruction>If the source describes a failure, lead with the failure signal (e.g. "Failed to …", "Error in …").</instruction>
  <instruction>Omit metadata that is stored separately (timestamps, IDs, scope keys).</instruction>
</instructions>

<quality_criteria>
  <criterion>A future vector/lexical search using related keywords MUST match this sentence.</criterion>
  <criterion>Reading this sentence alone should answer: "What happened and why does it matter?"</criterion>
</quality_criteria>

<output_format>
  <format>Single plain-text sentence. No wrapping tags in the output.</format>
</output_format>"""

MEMORY_HYBRID_PROMPT = """\
<task>Generate a medium-detail memory record optimised for context injection.</task>

<instructions>
  <instruction>Begin with a one-line summary (same quality bar as the compressed key).</instruction>
  <instruction>Follow with 2–4 bullet points capturing the most important supporting details.</instruction>
  <instruction>Each bullet should be a self-contained fact — not a sentence fragment.</instruction>
  <instruction>Include explicit status or result when the source contains one (success, failure, pending).</instruction>
  <instruction>Name all notable entities: tools, APIs, files, error codes, user-specified values.</instruction>
  <instruction>If the source contains a causal chain (A caused B), preserve it in one bullet.</instruction>
  <instruction>Omit boilerplate, greetings, and conversational filler from the source.</instruction>
</instructions>

<output_format>
  <summary>[one-line summary here]</summary>
  <details>
    - [key point 1]
    - [key point 2]
    - [key point 3 — optional]
    - [key point 4 — optional]
  </details>
</output_format>

<quality_criteria>
  <criterion>An agent reading only this record should understand the event well enough to avoid repeating mistakes or to build on prior progress.</criterion>
  <criterion>Total length should stay under 120 words.</criterion>
</quality_criteria>"""

MEMORY_FULL_PROMPT = """\
<task>Generate a full-context memory record preserving maximum evidential detail.</task>

<instructions>
  <instruction>Start with a concise summary line.</instruction>
  <instruction>Provide structured detail bullets covering all substantive information from the source.</instruction>
  <instruction>Include verbatim evidence snippets (quoted) when they carry diagnostic or decision-making value.</instruction>
  <instruction>Explicitly note any risks, failures, uncertainties, or open questions.</instruction>
  <instruction>End with a follow-up or next-step note if the source implies one; otherwise state "No follow-up required."</instruction>
  <instruction>Preserve numerical values, configuration parameters, and exact error messages.</instruction>
  <instruction>If multiple entities or tools are involved, list each with its role in the event.</instruction>
</instructions>

<output_format>
  <summary>[one-line summary]</summary>
  <structured_details>
    - [detail 1]
    - [detail 2]
    - [detail N]
  </structured_details>
  <evidence>
    - "[verbatim snippet or exact value]"
  </evidence>
  <risks_and_uncertainties>
    - [risk/uncertainty, or "None identified."]
  </risks_and_uncertainties>
  <next_steps>
    - [follow-up action, or "No follow-up required."]
  </next_steps>
</output_format>

<quality_criteria>
  <criterion>This record is the authoritative archive. Another agent with no prior context should be able to fully reconstruct what happened from this record alone.</criterion>
  <criterion>Prefer completeness over brevity — but never pad with filler.</criterion>
</quality_criteria>"""

RETRY_GUIDANCE_PROMPT = """\
<system>
<role>You are a Tool Failure Analyst inside an agentic memory system.</role>
<purpose>
  Analyse a failed tool invocation and produce concise, actionable guidance
  that prevents the agent from repeating the same mistake.
</purpose>
</system>

<instructions>
  <instruction>Identify the root cause category: bad arguments, missing dependency, permission error, timeout, rate limit, or unknown.</instruction>
  <instruction>State what specifically went wrong in one sentence.</instruction>
  <instruction>Provide 1-3 concrete corrective actions ranked by likelihood of success.</instruction>
  <instruction>If an alternative tool or approach exists, name it explicitly.</instruction>
  <instruction>Keep total output under 80 words — this will be injected into a live context window.</instruction>
</instructions>

<output_format>
  <diagnosis>[one-sentence root cause]</diagnosis>
  <corrective_actions>
    1. [most likely fix]
    2. [alternative approach — optional]
    3. [fallback — optional]
  </corrective_actions>
</output_format>

<anti_patterns>
  <anti_pattern>Do NOT suggest "try again with the same arguments" unless there is evidence of a transient failure.</anti_pattern>
  <anti_pattern>Do NOT produce vague advice like "check the documentation" — be specific.</anti_pattern>
</anti_patterns>"""

# ---------------------------------------------------------------------------
# Auto-planning prompts (memory-level intrinsic planning)
# ---------------------------------------------------------------------------
PLAN_COMPLEXITY_PROMPT = """\
<task>Decide whether a user request needs multi-step planning.</task>

<instructions>
  <instruction>Return "plan_required=true" when the request has multiple constraints, dependencies, or execution phases.</instruction>
  <instruction>Return "plan_required=false" when a direct single-step response is sufficient.</instruction>
  <instruction>Provide a one-line rationale and confidence score (0-1).</instruction>
</instructions>

<output_format>
plan_required=<true|false>
rationale=<one line>
confidence=<0.00-1.00>
</output_format>"""

PLAN_DECOMPOSITION_PROMPT = """\
<task>Create an executable short plan from a complex request.</task>

<instructions>
  <instruction>Generate 2-5 concrete steps in execution order.</instruction>
  <instruction>Each step must be action-oriented and verifiable.</instruction>
  <instruction>Include tool hints only when strongly implied by the request.</instruction>
</instructions>

<output_format>
goal=<single sentence>
steps:
1) <step text> | tools=<comma separated or none>
2) <step text> | tools=<comma separated or none>
...
confidence=<0.00-1.00>
</output_format>"""

PLAN_PROGRESS_PROMPT = """\
<task>Map execution evidence to plan step progression.</task>

<instructions>
  <instruction>Use tool outcomes and assistant output to infer whether a step is completed, blocked, or still in progress.</instruction>
  <instruction>Prefer explicit execution evidence over narrative claims.</instruction>
  <instruction>Return one step update at a time with a concise summary.</instruction>
</instructions>

<output_format>
step_index=<integer>
status=<completed|blocked|in_progress|failed>
summary=<one line>
confidence=<0.00-1.00>
blocked_reason=<one line or none>
</output_format>"""

PLAN_REPLAN_PROMPT = """\
<task>Produce a minimal replan suggestion when progress is blocked.</task>

<instructions>
  <instruction>Keep prior completed steps intact.</instruction>
  <instruction>Add at most one recovery step unless absolutely required.</instruction>
  <instruction>Prioritize concrete unblock actions over generic advice.</instruction>
</instructions>

<output_format>
should_replan=<true|false>
recovery_step=<text or none>
confidence=<0.00-1.00>
</output_format>"""

# ---------------------------------------------------------------------------
# Context-assembly section headers (used by ContextAssembler)
# ---------------------------------------------------------------------------
CONTEXT_MEMORY_HEADER = """\
<relevant_memories>
  The following memories were retrieved from long-term storage based on
  relevance to the current query. Use them to inform your response, but
  verify against the live conversation when details conflict.
</relevant_memories>"""

CONTEXT_PLAN_HEADER = """\
<active_plan>
  The user has an active multi-step plan. Track progress and align your
  actions with the current step. Update the plan status after completing
  or failing a step.
</active_plan>"""

CONTEXT_TOOL_GUIDANCE_HEADER = """\
<tool_guidance>
  Recent tool interaction history is provided below. Pay special attention
  to any failures or retry guidance — avoid repeating failed approaches.
</tool_guidance>"""

CONTEXT_COMPRESSED_HISTORY_HEADER = """\
<earlier_conversation>
  This is a compressed summary of earlier conversation turns that have
  been evicted from the active context window. Treat as background
  knowledge — details may be approximate.
</earlier_conversation>"""
