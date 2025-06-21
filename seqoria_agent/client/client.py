import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Iterable, List, NamedTuple, Optional, Union, Literal, AsyncGenerator, Callable
import logging

# NOTE: Replaced the heavy MCP StreamableHTTPTransport with a lean internal
# implementation that supports the subset of methods AIClient needs.
from .simple_http_transport import SimpleHTTPTransport as _HTTPTransport

from mcp.types import Content

# Try to import structlog for structured logging. Fallback to stdlib logger.
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – structlog optional
    structlog = None


# Provider abstraction layer
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall
from seqoria_agent.providers.errors import LLMError

from seqoria_agent.models.envelope import DisplayEnvelope
from seqoria_agent.models.errors import ToolRemoteError, CircuitOpenError, DependencyError
from seqoria_agent.client.parameters import ParameterProvider

from seqoria_agent.models.chunk import ChatChunk

# NEW: Synthetic tool names
THINK_TOOL_NAME = "__think__"
SUMMARIZE_TOOL_NAME = "__summarize__"
CLARIFY_TOOL_NAME = "__clarify__"

# Caching utilities
from .cache import CacheProtocol, DictCache, NullCache, build_cache_key

# Helper class for managing connections
class _ClientConnection(NamedTuple):
    """A container for an active FastMCP client and its URL."""
    client: _HTTPTransport
    url: str


class AIClient:
    """A unified AI client that orchestrates calls between LLMs and seqoria_agent servers.

    This client connects to one or more seqoria_agent/FastMCP servers, discovers the
    available tools, and uses a Large Language Model (LLM) to intelligently
    decide which tools to call based on a user's query. It supports iterative
    multi-turn conversations, parallel tool execution, response streaming,
    and various reliability and performance features like caching, retries,
    and circuit breakers.
    """

    def __init__(
        self,
        urls: Iterable[str],
        providers: List[BaseProvider],
        *,
        logger: Optional[logging.Logger] = None,
        on_error: Optional[Callable[[dict, list | None], None]] = None,
        on_display: Optional[Callable[[dict, list | None], None]] = None,
        refresh_interval: int | float = 0,
        param_providers: list["ParameterProvider"] | None = None,
        allow_override: bool = False,
        **kwargs: Any,
    ):
        """Initialise the AIClient.

        Parameters
        ----------
        urls : Iterable[str]
            A list of URLs for the MCP servers to connect to.
        providers : List[BaseProvider]
            A list of LLM provider instances (e.g., `OpenAIProvider`) to use
            for generating responses. They are tried in order.
        logger : Optional[logging.Logger], optional
            A custom logger instance. If None, a default `structlog` or
            standard library logger is used, by default None.
        on_error : Optional[Callable[[dict, list | None], None]], optional
            A synchronous callback invoked when a tool returns a structured
            error. The callback receives the error payload and the current
            conversation history, by default None.
        on_display : Optional[Callable[[dict, list | None], None]], optional
            A synchronous callback invoked when a display tool returns a
            `DisplayEnvelope`. This allows for immediate rendering before the
            `chat` method completes, by default None.
        refresh_interval : int | float, optional
            Interval in seconds to automatically refresh the tool registry from
            the servers. If 0, auto-refresh is disabled, by default 0.
        param_providers : list["ParameterProvider"] | None, optional
            List of parameter providers to use for populating tool parameters.
        allow_override : bool, optional
            Whether to allow overriding server-populated parameters with client-side defaults.
        **kwargs : Any
            Additional keyword arguments are passed to the underlying
            `fastmcp.Client` and can also be used to configure client features
            like caching, timeouts, and retry policies. See the documentation
            for a full list of available options.

        Raises
        ------
        ValueError
            If `urls` or `providers` are empty.
        """
        if not urls:
            raise ValueError("At least one URL must be provided.")
        self._urls = list(dict.fromkeys(urls))  # Remove duplicates

        # Validate providers list -------------------------------------------------
        if not providers:
            raise ValueError("At least one LLM provider instance must be supplied via the 'providers' argument.")

        self._providers: List[BaseProvider] = providers

        # Strip legacy OpenAI-specific kwargs so they don't leak into FastMCP
        for _legacy_key in ("openai_api_key", "llm_model", "fallback_models", "openai_base_url"):
            kwargs.pop(_legacy_key, None)

        # --------------------------------------------------------------
        # Caching setup – extract before passing remaining kwargs to FastMCP Client
        # --------------------------------------------------------------
        self._cache: CacheProtocol | None = kwargs.pop("cache", None)  # type: ignore[arg-type]
        # Cache TTL (seconds) can be overridden via kwarg; default 300 per README.
        self._cache_ttl: int | None = kwargs.pop("cache_ttl", 300)  # type: ignore[arg-type]

        # --------------------------------------------------------------
        # Adaptive concurrency & reliability parameters (NEW)
        # --------------------------------------------------------------

        max_concurrency = kwargs.pop("max_concurrency", None)
        self._tool_semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(value=int(max_concurrency)) if max_concurrency else None
        )

        self._tool_timeout: float | None = kwargs.pop("tool_timeout", None)

        self._retry_policy: dict[str, Any] = kwargs.pop("retry_policy", {}) or {}
        self._breaker_threshold: int = int(kwargs.pop("breaker_threshold", 5))
        self._breaker_reset_after: int = int(kwargs.pop("breaker_reset_after", 60))

        # Failure tracking for circuit breaker – keyed by server URL
        self._fail_log: dict[str, int] = {}
        self._breaker_until: dict[str, float] = {}

        # Ensure we always have a cache object to call (may be a no-op)
        if self._cache is None:
            self._cache = NullCache()

        # Remaining kwargs are forwarded to FastMCP Client constructor
        self._client_kwargs = kwargs

        self._connections: list[_ClientConnection] = []
        self._tool_map: dict[str, _HTTPTransport] = {}
        self._llm_tools: list[dict] = []
        # Mapping tool_name -> list[str] of server-populated parameters
        self._server_params: dict[str, list[str]] = {}
        # Track tools that can trigger an immediate display response
        self._display_tools: set[str] = set()

        # ------------------------------------------------------------------
        # Server-populated parameter providers (client side) -----------------
        # ------------------------------------------------------------------

        from seqoria_agent.client.parameters import ParameterProvider  # local import to avoid top-level cycles

        self._param_providers: list[ParameterProvider] = list(param_providers or [])  # type: ignore[arg-type]

        self._allow_override = bool(allow_override)

        # ------------------------------------------------------------------
        # Default system prompt – unchanged from original implementation
        # ------------------------------------------------------------------
        self._system_prompt = (
            "You are a helpful and intelligent assistant. You have access to a set of tools "
            "to answer user questions. When you receive a user's query, you must decide whether "
            "to call a tool or to respond directly. If you call a tool, you will receive the result "
            "and must then formulate a final, user-facing response based on that result. "
            "Always provide the final answer in a clear and friendly manner."
            "\nIf you need to reason through intermediate steps before the next tool, call the `__think__` tool with your thought."
            "\nIf the conversation has grown long, you may call the `__summarize__` tool to compress the history before continuing."
            "\nIf you need additional information from the user, call the `__clarify__` tool with a concise question."
        )

        # Structured logger – default to stdlib logger under 'seqoria_agent'
        if logger is not None:
            self._logger = logger
        else:
            if structlog is not None:
                self._logger = structlog.get_logger("seqoria_agent")  # type: ignore[assignment]
            else:
                self._logger = logging.getLogger(name="seqoria_agent")

        # Holds reference to *current* conversation history during a single
        # chat()/stream_chat() call. Reset to None immediately afterwards so
        # the client remains stateless between invocations.
        self._current_history: Optional[List[dict]] = None

        # Optional synchronous callback invoked whenever a structured error
        # envelope is received from a tool.
        self._on_error = on_error

        # Optional callback for successful DisplayEnvelope payloads (early-exit)
        self._on_display = on_display

        # --- Hot-reload of tool registry -----------------------------------
        self._refresh_interval: float = float(refresh_interval)
        self._refresher_task: Optional[asyncio.Task] = None
        self._tool_lock = asyncio.Lock()

        # Added for dependency graph support
        self._tool_deps: dict[str, set[str]] = {}

    async def __aenter__(self):
        connect_tasks = [self._connect_to_server(url) for url in self._urls]
        connections = await asyncio.gather(*connect_tasks)
        self._connections = [conn for conn in connections if conn is not None]

        if not self._connections:
            raise RuntimeError("Could not connect to any of the provided MCP servers.")

        await self._update_llm_tools()

        # Spawn periodic refresher if enabled
        if self._refresh_interval > 0:
            loop = asyncio.get_running_loop()
            self._refresher_task = loop.create_task(self._periodic_refresh())

        return self

    async def _connect_to_server(self, url: str) -> _ClientConnection | None:
        client = _HTTPTransport(url=url, **self._client_kwargs)
        try:
            await client.__aenter__()
            self._logger.info("Connected to MCP server", url=url)
            return _ClientConnection(client=client, url=url)
        except Exception as e:
            self._logger.warning("Failed to connect to MCP server", url=url, error=str(e))
            await client.close()
            return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Gracefully close all server connections and background tasks.

        This method should be called to ensure a clean shutdown of the client.
        It is called automatically when using the client in an `async with`
        statement.
        """
        exit_tasks = [conn.client.close() for conn in self._connections]
        await asyncio.gather(*exit_tasks, return_exceptions=True)
        self._connections.clear()
        self._tool_map.clear()

        # Cancel refresher task if running
        if self._refresher_task and not self._refresher_task.done():
            self._refresher_task.cancel()
            try:
                await self._refresher_task
            except Exception:
                pass

    async def _update_llm_tools(self):
        """Fetch and merge tools from all servers (thread-safe)."""

        async with self._tool_lock:
            self._tool_map.clear()
            self._display_tools.clear()
            # Reset dependency mapping each time we refresh the tool catalogue
            self._tool_deps.clear()

        tool_results = await asyncio.gather(
            *(conn.client.list_tools() for conn in self._connections),
            return_exceptions=True,
        )

        llm_tools = []
        for i, result in enumerate(tool_results):
            conn = self._connections[i]
            if isinstance(result, Exception):
                self._logger.warning("Could not list tools from server", url=conn.url, error=str(result))
                continue
            
            for tool in result:
                if tool.name not in self._tool_map:
                    self._tool_map[tool.name] = conn.client

                    # Check for our special tool annotations
                    if tool.annotations:
                        # ``tool.annotations`` may be either a plain ``dict`` (FastMCP <=0.4)
                        # or a *pydantic* model (>=0.5).  Handle both cases gracefully.

                        if isinstance(tool.annotations, dict):
                            if tool.annotations.get("is_display_tool"):
                                self._display_tools.add(tool.name)

                            sp = tool.annotations.get("server_populated") or []
                            if isinstance(sp, list):
                                self._server_params[tool.name] = list(sp)

                            # NEW – dependency graph annotation
                            dep = tool.annotations.get("depends_on") or []
                            if isinstance(dep, list):
                                self._tool_deps[tool.name] = set(dep)
                        else:
                            # Attempt attribute access; fall back to ``model_dump`` if available.
                            is_display = False
                            if hasattr(tool.annotations, "is_display_tool"):
                                is_display = bool(getattr(tool.annotations, "is_display_tool"))
                            elif hasattr(tool.annotations, "model_dump"):
                                try:
                                    is_display = bool(tool.annotations.model_dump().get("is_display_tool"))
                                except Exception:
                                    is_display = False

                            if is_display:
                                self._display_tools.add(tool.name)

                            # Try to extract server_populated field list
                            sp_list: list[str] | None = None
                            if hasattr(tool.annotations, "server_populated"):
                                sp_list = getattr(tool.annotations, "server_populated")
                            elif hasattr(tool.annotations, "model_dump"):
                                try:
                                    sp_list = tool.annotations.model_dump().get("server_populated")
                                except Exception:
                                    sp_list = None

                            if sp_list:
                                self._server_params[tool.name] = list(sp_list)

                            # Attempt to extract depends_on from pydantic annotation model
                            dep_list: list[str] | None = None
                            if hasattr(tool.annotations, "depends_on"):
                                dep_list = getattr(tool.annotations, "depends_on")
                            elif hasattr(tool.annotations, "model_dump"):
                                try:
                                    dep_list = tool.annotations.model_dump().get("depends_on")
                                except Exception:
                                    dep_list = None

                            if dep_list:
                                self._tool_deps[tool.name] = set(dep_list)

                    # Format for OpenAI tool-calling
                    llm_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    })
        
        # --- Synthetic "__think__" tool --------------------------------------
        llm_tools.append({
            "type": "function",
            "function": {
                "name": THINK_TOOL_NAME,
                "description": "Record internal reasoning steps without showing them to the user. Use before the next tool call if you need to think.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Internal chain-of-thought. Not shown to the user."}
                    },
                    "required": ["thought"],
                },
            },
        })
        # ---------------------------------------------------------------------

        # --- Synthetic "__summarize__" tool ----------------------------------
        llm_tools.append({
            "type": "function",
            "function": {
                "name": SUMMARIZE_TOOL_NAME,
                "description": "Summarize the conversation so far, replacing earlier history with the summary to save tokens.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        })
        # ---------------------------------------------------------------------

        # --- Synthetic "__clarify__" tool ------------------------------------
        llm_tools.append({
            "type": "function",
            "function": {
                "name": CLARIFY_TOOL_NAME,
                "description": "Ask the user a follow-up question when instructions are ambiguous.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "A single clarifying question for the user."}
                    },
                    "required": ["question"],
                },
            },
        })
        # ---------------------------------------------------------------------

        # Atomically replace tool list
        async with self._tool_lock:
            self._llm_tools = llm_tools

        self._logger.info("AI Client has discovered tools", count=len(llm_tools))

        # Done – dependency info has been captured in ``self._tool_deps``

    async def call_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        server_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Content]:
        """Call a specific tool on an MCP server with retry and circuit-breaker logic.

        This is a low-level method for directly invoking a tool. In most cases,
        you should use the `chat` or `stream_chat` methods, which handle tool
        calling automatically.

        Parameters
        ----------
        name : str
            The name of the tool to call.
        arguments : dict[str, Any]
            The arguments to pass to the tool.
        server_args : dict[str, Any] | None, optional
            Additional arguments to pass to the tool on the server side.
        **kwargs : Any
            Additional keyword arguments to pass to the `fastmcp.Client.call_tool`
            method.

        Returns
        -------
        list[Content]
            The result from the tool, typically a list containing a single
            `Content` object.

        Raises
        ------
        ValueError
            If the tool `name` is not found on any connected server.
        CircuitOpenError
            If the circuit breaker for the target server is currently open.
        ToolRemoteError
            If the tool call fails after all retry attempts.
        """

        client = self._tool_map.get(name)
        if not client:
            raise ValueError(f"Tool '{name}' not found on any connected server.")

        # Resolve server URL for tracking
        server_url: str | None = None
        for conn in self._connections:
            if conn.client is client:
                server_url = conn.url
                break
        server_url = server_url or "<unknown>"

        # ----- Circuit breaker gate -----------------------------------
        now = time.time()
        if self._breaker_until.get(server_url, 0) > now:
            raise CircuitOpenError(f"Circuit open for server {server_url} – retry later.")

        # Retry config
        max_attempts: int = int(self._retry_policy.get("max_attempts", 1))
        backoff_base: float = float(self._retry_policy.get("backoff_base", 0.5))

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = await client.call_tool(
                    name=name,
                    arguments=arguments,
                    server_args=server_args or {},
                    **kwargs,
                )
                # Success – clear failure count
                self._fail_log[server_url] = 0
                return result
            except Exception as exc:
                # Register failure and maybe backoff / retry
                self._fail_log[server_url] = self._fail_log.get(server_url, 0) + 1

                if attempt >= max_attempts:
                    # Threshold reached? Trip breaker.
                    if self._fail_log[server_url] >= self._breaker_threshold:
                        self._breaker_until[server_url] = now + self._breaker_reset_after
                    raise ToolRemoteError(exc) from exc

                # Back-off before next attempt
                await asyncio.sleep(delay=backoff_base * (2 ** (attempt - 1)))

        # This point shouldn't be reached, but mypy happy
        raise ToolRemoteError("Tool call failed after retries.")

    async def chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
    ) -> dict[str, Any]:
        """Engage in a stateless, multi-turn chat conversation.

        This method orchestrates the conversation flow: it sends the history
        to the LLM, executes any requested tools in parallel, and loops until
        the LLM provides a final, user-facing answer.

        The client itself is stateless. The caller is responsible for
        persisting the `history` list between calls.

        Parameters
        ----------
        user_query : str
            The user's latest message for this turn.
        history : List[dict]
            The conversation history, a list of message dictionaries in the
            OpenAI format. This list will be mutated in-place.
        max_turns : int, optional
            The maximum number of LLM-tool iterations to perform before
            stopping, as a safety measure to prevent infinite loops,
            by default 5.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the result:
            - "answer" (`str`): The final text response from the LLM.
            - "display" (`dict` | `None`): A `DisplayEnvelope` if a display
              tool was called and triggered an early exit.
            - "history" (`list[dict]`): The complete, updated conversation
              history.

        Raises
        ------
        ValueError
            If `history` is `None`.
        RuntimeError
            If the method is called re-entrantly.
        """

        if history is None:  # Defensive (typing promises non-None)
            raise ValueError("'history' must be provided to AIClient.chat().")

        # Guard: prevent accidental re-entrancy that would corrupt state.
        if self._current_history is not None:
            raise RuntimeError("AIClient.chat() re-entered while a previous call is still running.")

        self._current_history = history

        # 1️⃣ Ensure system prompt at index 0
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self._system_prompt})

        # 2️⃣ Append the user message for this turn
        history.append({"role": "user", "content": user_query})

        # ------------------------------------------------------------------
        # Main LLM ➜ Tool loop
        # ------------------------------------------------------------------
        try:
            for turn in range(max_turns):
                self._logger.info("LLM reasoning", turn=turn + 1, max_turns=max_turns)

                llm_resp = await self._call_llm(messages=history)

                # Build assistant message dict for conversation history ----------
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_resp.text,
                }

                if llm_resp.tool_calls:
                    assistant_msg["tool_calls"] = [tc.model_dump() for tc in llm_resp.tool_calls]

                history.append(assistant_msg)

                # 🤔 LLM produced a plain message (no tool calls). We treat this as the final answer and exit.
                if not llm_resp.tool_calls:
                    self._logger.info("LLM responded directly - conversation resolved without tool calls.")
                    return {"answer": llm_resp.text, "display": None, "history": history}

                # 🛠️ LLM requested one or more tool calls.
                tool_calls = llm_resp.tool_calls

                # ------------------------------------------------------------------
                # 3. Dependency-aware Tool Execution 🚀
                # ------------------------------------------------------------------
                try:
                    layers = self._compute_execution_layers(tool_calls)
                except DependencyError as de:
                    # Surface dependency errors to user & stop.
                    self._logger.error("Dependency resolution failed", error=str(de))
                    return {"answer": f"[DependencyError] {de}", "display": None, "history": history}

                early_exit_payload: Optional[dict[str, Any]] = None

                for layer in layers:
                    coros = [self._run_tool(tc) for tc in layer]
                    results = await asyncio.gather(*coros, return_exceptions=False)

                    for (_, history_entry, maybe_payload) in results:
                        if history_entry:
                            history.append(history_entry)

                        if early_exit_payload is None and maybe_payload is not None:
                            early_exit_payload = maybe_payload

                    # Early-exit short-circuits remaining layers
                    if early_exit_payload is not None:
                        early_exit_payload["history"] = history
                        return early_exit_payload

                # → Loop continues if no early exit – LLM will see all tool outputs

            # Safety guard hit – give up gracefully
            self._logger.warning("Reached max_turns without a final answer.")
            return {
                "answer": "[Turn limit reached without resolution]",
                "display": None,
                "history": history,
            }
        finally:
            # Ensure statelessness after each call
            self._current_history = None

    # ------------------------------------------------------------------
    # 🚀 NEW – Streaming chat API
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        user_query: str,
        history: List[dict],
        *,
        max_turns: int = 5,
    ) -> AsyncGenerator[ChatChunk, None]:
        """Stream the chat response as it's being generated.

        This method provides a real-time, token-by-token stream of the LLM's
        output and yields structured `DisplayEnvelope` payloads as soon as
        they are returned by tools. It follows the same iterative logic as
        the `chat` method.

        Parameters
        ----------
        user_query : str
            The user's latest message for this turn.
        history : List[dict]
            The conversation history, which will be mutated in-place.
        max_turns : int, optional
            The maximum number of LLM-tool iterations, by default 5.

        Yields
        ------
        AsyncGenerator[ChatChunk, None]
            An asynchronous generator of `ChatChunk` objects. Each chunk has
            a `type` of either `"llm_delta"` (containing new text) or
            `"tool_display"` (containing a `DisplayEnvelope`).

        Raises
        ------
        ValueError
            If `history` is `None`.
        RuntimeError
            If the method is called re-entrantly or if all LLM providers fail.
        """

        if history is None:
            raise ValueError("'history' must be provided to AIClient.stream_chat().")

        # Guard against re-entrancy
        if self._current_history is not None:
            raise RuntimeError("AIClient.stream_chat() re-entered while a previous call is still running.")

        self._current_history = history

        # Ensure system prompt present
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self._system_prompt})

        # Append user message
        history.append({"role": "user", "content": user_query})

        try:
            for turn in range(max_turns):
                self._logger.info("LLM streaming", turn=turn + 1, max_turns=max_turns)

                # ----------------------------------------------------------
                # Select first provider that starts streaming successfully
                # ----------------------------------------------------------
                stream_gen: Optional[AsyncGenerator[Union[str, LLMResponse], None]] = None
                chosen_provider: Optional[BaseProvider] = None
                last_err: Optional[Exception] = None

                for provider in self._providers:
                    try:
                        stream_gen = provider.async_stream(
                            chat_history=history,
                            tools=self._llm_tools,
                            tool_choice="auto",
                        )
                        # Try to aclose later; assume success
                        chosen_provider = provider
                        break
                    except LLMError as e:
                        self._logger.warning(
                            "Provider failed to start stream – trying next",
                            provider=provider.__class__.__name__,
                            error=str(e),
                        )
                        last_err = e
                        continue

                if stream_gen is None:
                    raise RuntimeError(f"All LLM providers failed to stream. Last error: {last_err}")

                # ----------------------------------------------------------
                # Consume the stream and forward deltas
                # ----------------------------------------------------------
                aggregated_text: List[str] = []
                final_response: Optional[LLMResponse] = None

                async for chunk in stream_gen:
                    if isinstance(chunk, str):
                        aggregated_text.append(chunk)
                        yield ChatChunk(type="llm_delta", text=chunk)
                    else:
                        # Accept either LLMResponse or plain dict for backwards compatibility
                        if isinstance(chunk, LLMResponse):
                            final_response = chunk
                        elif isinstance(chunk, dict):
                            try:
                                final_response = LLMResponse.model_validate(chunk)
                            except Exception:
                                # Fallback to minimal conversion
                                final_response = LLMResponse(
                                    model_id=chunk.get("model_id", "unknown"),
                                    text=chunk.get("text"),
                                    tool_calls=[ToolCall.model_validate(tc) if not isinstance(tc, ToolCall) else tc for tc in (chunk.get("tool_calls") or [])],
                                    usage=chunk.get("usage"),
                                    stop_reason=chunk.get("stop_reason"),
                                )
                        else:
                            # Unexpected data type – ignore but log
                            self._logger.warning("Unknown chunk type from provider stream", type=str(type(chunk)))

                # If provider didn't emit a final LLMResponse, fabricate one
                if final_response is None:
                    final_response = LLMResponse(
                        model_id="unknown",
                        text="".join(aggregated_text),
                    )

                # Update history ------------------------------------------------
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": final_response.text,
                }
                if final_response.tool_calls:
                    assistant_msg["tool_calls"] = [tc.model_dump() for tc in final_response.tool_calls]

                history.append(assistant_msg)

                # No tool calls → we're done
                if not final_response.tool_calls:
                    return

                # Execute requested tools with dependency graph --------------
                try:
                    layers = self._compute_execution_layers(final_response.tool_calls)
                except DependencyError as de:
                    self._logger.error("Dependency resolution failed (stream mode)", error=str(de))
                    yield ChatChunk(type="llm_delta", text=f"[DependencyError] {de}")
                    return

                early_exit_payload: Optional[dict[str, Any]] = None

                for layer in layers:
                    coros = [self._run_tool(tc) for tc in layer]
                    results = await asyncio.gather(*coros, return_exceptions=False)

                    for (_, history_entry, maybe_payload) in results:
                        if history_entry:
                            history.append(history_entry)

                        if early_exit_payload is None and maybe_payload is not None:
                            early_exit_payload = maybe_payload

                    if early_exit_payload is not None:
                        if early_exit_payload.get("display") is not None:
                            yield ChatChunk(type="tool_display", envelope=early_exit_payload["display"])
                        if early_exit_payload.get("answer"):
                            yield ChatChunk(type="llm_delta", text=early_exit_payload["answer"])
                        return

                # → Loop continues with updated history

            # Safety guard hit
            self._logger.warning("Reached max_turns without a final answer (stream mode).")
            yield ChatChunk(type="llm_delta", text="[Turn limit reached without resolution]")
        finally:
            self._current_history = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_tool(
        self,
        tool_call: ToolCall,
    ) -> tuple[str | None, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Execute a single tool and *return* the history entry instead of
        mutating shared state.

        Returns a tuple ``(tool_call_id, history_entry, early_exit_payload)``.
        ``history_entry`` is the dict to be appended to conversation history
        (or ``None`` if an early-exit occurred). The *early_exit_payload*
        mirrors the structure returned by :meth:`chat` when a display-only or
        hybrid display path is taken.
        """

        func_dict = tool_call.function or {}
        tool_name = func_dict.get("name")
        try:
            arg_payload = func_dict.get("arguments", {})
            if isinstance(arg_payload, str):
                arguments = json.loads(arg_payload)
            elif isinstance(arg_payload, dict):
                arguments = arg_payload
            else:
                arguments = {}
        except Exception as e:
            arguments = {}
            self._logger.warning("Could not parse arguments for tool", tool=tool_name, error=str(e))

        # ------------------------------------------------------------------
        # 🔄  Client-side server-populated parameter injection -------------
        # ------------------------------------------------------------------

        server_args: dict[str, Any] = {}

        spp_fields = self._server_params.get(tool_name, [])
        if spp_fields:
            from seqoria_agent.client.parameters import ParamContext

            # Remove AI-provided values when override disallowed
            if not self._allow_override:
                for fld in spp_fields:
                    arguments.pop(fld, None)

            # Determine which params need injection
            missing = [fld for fld in spp_fields if fld not in arguments]

            if missing and self._param_providers:
                ctx: ParamContext = {
                    "history": self._current_history or [],
                }

                for fld in list(missing):
                    for prov in self._param_providers:
                        try:
                            res = prov.resolve(fld, tool_name, ctx)  # type: ignore[attr-defined]
                            if asyncio.iscoroutine(res):
                                res = await res  # type: ignore[assignment]
                        except Exception:
                            continue

                        if res is not None:
                            server_args[fld] = res
                            break

            # If still missing after providers, error
            still_missing = [fld for fld in spp_fields if fld not in arguments and fld not in server_args]
            if still_missing:
                raise RuntimeError(
                    f"Missing server-populated parameters for tool '{tool_name}': {', '.join(still_missing)}"
                )

        # NEW: Handle the synthetic "__think__" tool for internal reasoning
        if tool_name == THINK_TOOL_NAME:
            thought_text = arguments.get("thought", "")

            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": thought_text,
            }
            self._logger.info("LLM thinking step recorded")
            return (tool_call.id, history_entry, None)

        # NEW: Handle summary tool
        if tool_name == SUMMARIZE_TOOL_NAME:
            summary_text = await self._summarize_history()
            if self._current_history and self._current_history[0].get("role") == "system":
                self._current_history[1:] = [{"role": "assistant", "content": summary_text}]
            else:
                self._current_history[:] = [{"role": "assistant", "content": summary_text}]

            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": "conversation summarized",
            }
            self._logger.info("Conversation summarized and history compressed")
            return (tool_call.id, history_entry, None)

        # Clarify tool
        if tool_name == CLARIFY_TOOL_NAME:
            question_text = arguments.get("question", "")
            history_entry = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": question_text,
            }
            early_payload = {
                "answer": question_text,
                "display": None,
                "history": None,
            }
            self._logger.info("Clarification requested from user")
            return (tool_call.id, history_entry, early_payload)

        # -------------------- Cache lookup ------------------------------
        cache_key = build_cache_key(tool_name=tool_name, arguments=arguments)
        show_user = arguments.get("show_user")
        is_display_call = tool_name in self._display_tools and bool(show_user)

        cached_output: Any | None = None
        try:
            if self._cache is not None:
                cached_output = await self._cache.get(key=cache_key)
                if cached_output is not None:
                    self._logger.info("Cache hit - skipping tool execution")
        except Exception as e:
            self._logger.warning("Cache error (ignored)", error=str(e))

        if cached_output is not None:
            raw_output = cached_output
            display_envelope: Optional[dict[str, Any]] = None
            if isinstance(raw_output, (dict, list)):
                display_envelope = raw_output
        else:
            self._logger.info("Executing tool", tool=tool_name, args=arguments)

            try:
                async with (self._tool_semaphore or _null_async_cm()):
                    exec_coro = self.call_mcp_tool(name=tool_name, arguments=arguments, server_args=server_args)
                    if self._tool_timeout:
                        exec_coro = asyncio.wait_for(fut=exec_coro, timeout=self._tool_timeout)
                    tool_result = await exec_coro
            except asyncio.TimeoutError:
                raw_output = (
                    f"Tool '{tool_name}' timed out after {self._tool_timeout}s"
                )
                display_envelope = None
            except CircuitOpenError as e:
                raw_output = f"Circuit open - {e}"
                display_envelope = None
            except ToolRemoteError as e:
                raw_output = f"Error executing tool: {e}"
                display_envelope = None
            else:
                if tool_result and hasattr(tool_result[0], "text"):
                    raw_output = tool_result[0].text  # type: ignore[assignment]
                else:
                    raw_output = "Tool executed successfully with no textual output."

                display_envelope = None
                if isinstance(raw_output, (dict, list)):
                    display_envelope = raw_output
                else:
                    try:
                        display_envelope = json.loads(raw_output)
                    except Exception:
                        pass

                if display_envelope is not None and not isinstance(display_envelope, DisplayEnvelope):
                    try:
                        display_envelope = DisplayEnvelope.model_validate(display_envelope).dict_for_transport()
                    except Exception:
                        pass

                try:
                    if self._cache is not None:
                        await self._cache.set(key=cache_key, value=raw_output, ttl=self._cache_ttl)
                except Exception as e:
                    self._logger.warning("Cache set error (ignored)", error=str(e))

        # ------------------------------------------------------------------
        # 🛑 Structured error detection (NEW)
        # ------------------------------------------------------------------
        error_payload: Optional[dict[str, Any]] = None
        if isinstance(raw_output, str):
            error_payload = self._extract_error_from_xml(raw_output)

        history_entry = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": raw_output,
        }

        if is_display_call:
            self._logger.info("Display tool completed - will trigger early exit")
            answer_text = f"[Action '{tool_name}' was successful.]"
            early_payload = {
                "answer": answer_text,
                "display": display_envelope,
                "history": None,
            }

            # Invoke on_display callback synchronously if configured
            if display_envelope is not None and self._on_display is not None:
                try:
                    self._on_display(display_envelope, self._current_history)
                except Exception as cb_exc:  # pragma: no cover – defensive
                    self._logger.warning("on_display callback failed", error=str(cb_exc))

            return (tool_call.id, history_entry, early_payload)

        # Structured error → *no* early exit anymore. Instead, invoke the
        # optional on_error callback so host applications can react, and let
        # the LLM decide the next step with the XML embedded in history.

        if error_payload is not None and hasattr(self, "_on_error") and self._on_error:
            try:
                self._on_error(error_payload, self._current_history)
            except Exception as cb_exc:  # pragma: no cover – defensive
                self._logger.warning("on_error callback failed", error=str(cb_exc))

        return (tool_call.id, history_entry, None)

    async def _call_llm(self, messages: List[dict]) -> LLMResponse:
        """Invoke the **first** provider that successfully returns an `LLMResponse`.

        The method iterates over the list of providers passed during
        construction and calls :py:meth:`BaseProvider.async_generate` on each
        until one succeeds.  If all providers fail with an :class:`LLMError`
        the *last* error is raised inside a :class:`RuntimeError` to preserve
        backwards-compatible semantics.
        """

        last_err: Optional[Exception] = None

        for provider in self._providers:
            try:
                raw_resp = await provider.async_generate(
                    chat_history=messages,
                    tools=self._llm_tools,
                    tool_choice="auto",
                )

                # Coerce plain dict responses into LLMResponse models for uniformity
                if isinstance(raw_resp, LLMResponse):
                    response = raw_resp
                else:
                    try:
                        response = LLMResponse.model_validate(raw_resp)  # type: ignore[arg-type]
                    except Exception:
                        # Fallback minimal wrapping
                        response = LLMResponse(
                            model_id=getattr(raw_resp, "model_id", "unknown"),
                            text=getattr(raw_resp, "text", None) if isinstance(raw_resp, dict) else None,
                            tool_calls=getattr(raw_resp, "tool_calls", None) if isinstance(raw_resp, dict) else None,
                            usage=getattr(raw_resp, "usage", None) if isinstance(raw_resp, dict) else None,
                            stop_reason=getattr(raw_resp, "stop_reason", None) if isinstance(raw_resp, dict) else None,
                        )

                self._logger.info(
                    "LLM call successful",
                    provider=provider.__class__.__name__,
                    model=getattr(response, "model_id", "<unknown>"),
                )
                return response
            except LLMError as e:
                self._logger.warning(
                    "LLM provider failed, trying next",
                    provider=provider.__class__.__name__,
                    error=str(e),
                )
                last_err = e
                continue
            except Exception as e:  # pragma: no cover – unexpected
                # Don't hide non-LLM errors; re-raise immediately.
                raise

        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

    # ------------------------------------------------------------------
    # NEW – Structured error XML parsing helper
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_error_from_xml(xml_str: str) -> Optional[dict[str, Any]]:  # noqa: D401
        """Parse *xml_str* and return error payload dict or ``None`` if not an error."""

        if "<error" not in xml_str:
            return None

        try:
            import xml.etree.ElementTree as ET
            import json as _json

            root = ET.fromstring(xml_str.strip())
            error_el = root.find("error")
            if error_el is None:
                return None

            code = error_el.attrib.get("code", "UNKNOWN")
            retryable = error_el.attrib.get("retryable", "false").lower() == "true"

            # Text may be mixed: retrieve CDATA / text
            detail_text = (error_el.text or "").strip()

            meta_node = error_el.find("meta")
            meta: Any | None = None
            if meta_node is not None and meta_node.text:
                try:
                    meta = _json.loads(meta_node.text)
                except Exception:
                    meta = meta_node.text.strip()

            return {
                "code": code,
                "retryable": retryable,
                "detail": detail_text,
                "meta": meta,
            }
        except Exception:
            # Malformed XML – ignore
            return None

    # ------------------------------------------------------------------
    # 🚧  Placeholder – Conversation summarisation helper
    # ------------------------------------------------------------------
    async def _summarize_history(self) -> str:
        """Return a lightweight summary of the current conversation history.

        The original implementation delegated summarisation to the LLM via a
        dedicated tool call.  For the purposes of this refactor we provide a
        *very* naive fallback that simply truncates the existing history and
        returns the last few user/assistant messages joined together.  This
        prevents attribute errors until a more sophisticated approach is
        re-implemented.
        """

        if not self._current_history:
            return "(no history)"

        # Use the last N messages as a crude summary.
        N = 6
        recent_msgs = self._current_history[-N:]
        parts: list[str] = []
        for msg in recent_msgs:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API – hot-reload support
    # ------------------------------------------------------------------

    async def refresh_tools(self) -> None:
        """Manually trigger an update of the available tool catalogue.

        This method connects to all configured servers, fetches their current
        list of tools, and updates the client's internal tool registry.
        """

        await self._update_llm_tools()

    # Background refresher task -----------------------------------------

    async def _periodic_refresh(self) -> None:
        """Run :pyfunc:`refresh_tools` every :attr:`_refresh_interval` seconds."""

        try:
            while True:
                await asyncio.sleep(self._refresh_interval)
                try:
                    await self.refresh_tools()
                except Exception as exc:  # pragma: no cover – keep background alive
                    self._logger.warning("Background tool refresh failed", error=str(exc))
        except asyncio.CancelledError:
            # Normal shutdown
            return

    # ------------------------------------------------------------------
    # Dependency graph utilities
    # ------------------------------------------------------------------

    def _compute_execution_layers(self, tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Return topologically-sorted execution layers for *tool_calls*.

        The method raises :class:`DependencyError` when cycles are detected.
        Only dependencies *within* the current turn are enforced – if a
        predecessor is not part of *tool_calls* we assume it was executed in
        a previous turn and treat it as already satisfied.
        """

        from collections import defaultdict, deque
        from seqoria_agent.models.errors import DependencyError

        call_map: dict[str, ToolCall] = {
            (tc.function or {}).get("name"): tc for tc in tool_calls
        }

        requested: set[str] = set(call_map.keys())

        # Build graph (edges: dep -> tool)
        graph: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = {name: 0 for name in requested}

        for tool_name in requested:
            deps = self._tool_deps.get(tool_name, set())
            for dep in deps:
                if dep in requested:
                    graph[dep].add(tool_name)
                    indegree[tool_name] += 1

        # Kahn ‑ produce layers
        layers: list[list[ToolCall]] = []
        queue: deque[str] = deque([n for n, d in indegree.items() if d == 0])
        processed = 0

        while queue:
            current_layer: list[ToolCall] = []
            for _ in range(len(queue)):
                n = queue.popleft()
                processed += 1
                current_layer.append(call_map[n])
                for succ in graph.get(n, []):
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        queue.append(succ)
            layers.append(current_layer)

        if processed != len(requested):
            raise DependencyError("Cyclic dependency detected between requested tool calls.")

        return layers

# ---------------------------------------------------------------------------
# Utility – no-op async context manager (used when no semaphore limit set)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _null_async_cm():
    yield