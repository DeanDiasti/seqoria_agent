from __future__ import annotations

"""Provider interaction utilities for *seqoria_agent*'s new AIClient implementation.

This module is introduced in **Phase 4 - LLMInterface extraction** (see
``client-migration.md``) and encapsulates *all* logic that interacts directly
with Large-Language-Model providers.

Public façade
-------------
```
llm = LLMInterface(providers=[...])
resp = await llm.generate(history, tools=[...])
async for chunk in llm.stream(history):
    ...
```

The class keeps a **narrow** surface so that alternative provider strategies or
advanced fall-back logic can evolve without touching the surrounding
`AIClient` orchestration layer.
"""

from typing import Any, AsyncGenerator, List, Optional, Union

import logging
import asyncio

# Prefer structlog when available for key-value logging like elsewhere in the codebase
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None

from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList
from seqoria_agent.models.chunk import AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult, ChatChunk, ToolCallRequestList, LLMUsage

__all__: list[str] = ["LLMInterface"]


class LLMInterface:  # noqa: D101 – module-level docstring provides context
    # LOC gate: keep implementation short (< 250 lines)

    def __init__(self, 
                 providers: List[AsyncBaseProvider], 
                 *, 
                 logger: Optional[logging.Logger] = None) -> None:
        if not providers:
            raise ValueError("'providers' must contain at least one AsyncBaseProvider instance.")
        self._providers: List[AsyncBaseProvider] = providers
        # Cache index of provider that succeeded last time – optimistic fast-path
        self._primary_provider_idx: int = 0
        if logger is not None:
            self._logger = logger
        else:
            # Use stdlib logging to avoid structlog API differences during testing
            self._logger = logging.getLogger(name="seqoria_agent.llm_interface")

    # ------------------------------------------------------------------
    # Public API – synchronous generation
    # ------------------------------------------------------------------

    async def generate(self,
                       history: List[Any],
                       *,
                       tools: List[dict] | None = None,
                       tool_choice: str = "auto",
                       model: str | None = None) -> Union[AssistantMessage, ToolCallRequestList]:
        """Generate a single response using the first provider that succeeds.

        Parameters
        ----------
        history
            The chat history in OpenAI format (mutated by the caller).
        tools, tool_choice
            Optional tool specification forwarded to the provider.

        Returns
        -------
        AssistantMessage or ToolCallRequestList
            The model's answer.

        Raises
        ------
        RuntimeError
            If *all* providers fail with :class:`LLMError`.
        """
        kwargs_gen: dict[str, Any] = {"chat_history": history}

        kwargs_gen["tools"] = tools if tools is not None else None   
        kwargs_gen["tool_choice"] = (tool_choice or "auto") if tools else None
        kwargs_gen["model"] = model
            
        last_err: Optional[Exception] = None

        # Build provider iteration order: current primary first, then the rest.
        provider_indices = list(range(len(self._providers)))
        if self._primary_provider_idx in provider_indices:
            provider_indices.remove(self._primary_provider_idx)
        provider_indices.insert(0, self._primary_provider_idx)

        raw_resp: Optional[Union[LLMResponse, ToolCallList]] = None
        successful_idx: Optional[int] = None

        for idx in provider_indices:
            provider = self._providers[idx]
            try:
                raw_resp = await provider.async_generate(**kwargs_gen)
                successful_idx = idx
                break  # success
            except LLMError as e:
                last_err = e
                self._logger.warning("Provider %s failed to generate: %s", provider.__class__.__name__, e)
                continue
            except Exception as e:
                # Non-LLM errors are not considered retryable – re-raise.
                self._logger.warning("Provider %s failed to generate: %s", provider.__class__.__name__, e)
                raise

        if raw_resp is None:
            raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

        # Update fast-path cache on success
        if successful_idx is not None:
            self._primary_provider_idx = successful_idx

        if isinstance(raw_resp, LLMResponse):
            usage = raw_resp.usage or LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return AssistantMessage(role="assistant",
                                    content=raw_resp.text,
                                    model=raw_resp.model_id,
                                    usage=LLMUsage(prompt_tokens=usage.prompt_tokens,
                                                   completion_tokens=usage.completion_tokens,
                                                   total_tokens=usage.total_tokens))

        if isinstance(raw_resp, ToolCallList):
            return ToolCallRequestList(tool_calls=[ToolCallRequest(role="tool_call",
                                                                   id=tool_call.id,
                                                                   type="function_call",
                                                                   name=tool_call.function_name,
                                                                   arguments=tool_call.arguments) for tool_call in raw_resp.tool_calls],
                                       usage=LLMUsage(prompt_tokens=raw_resp.usage.prompt_tokens,
                                                      completion_tokens=raw_resp.usage.completion_tokens,
                                                      total_tokens=raw_resp.usage.total_tokens))


    # ------------------------------------------------------------------
    # Public API – streaming generation
    # ------------------------------------------------------------------

    async def stream(self,
                     history: List[Any],
                     *,
                     tools: List[dict] | None = None,
                     tool_choice: str = "auto",
                     model: str | None = None) -> AsyncGenerator[Union[ChatChunk, AssistantMessage, ToolCallRequestList], None]:
        """Yield incremental deltas followed by the final :class:`LLMResponse`.

        The method attempts each provider in order until a stream generator is
        obtained.  It proxies every delta (``str``) downstream immediately and
        finally yields the *full* ``LLMResponse`` once the provider signals
        completion.
        """

        kwargs_stream: dict[str, Any] = {"chat_history": history}

        kwargs_stream["tools"] = tools if tools is not None else None   
        kwargs_stream["tool_choice"] = (tool_choice or "auto") if tools else None
        kwargs_stream["model"] = model

        stream_gen: Optional[AsyncGenerator[Union[str, ChatChunk, AssistantMessage, ToolCallRequestList], None]] = None
        last_err: Optional[Exception] = None

        provider_indices = list(range(len(self._providers)))
        if self._primary_provider_idx in provider_indices:
            provider_indices.remove(self._primary_provider_idx)
        provider_indices.insert(0, self._primary_provider_idx)

        raw_resp: Optional[Union[LLMResponse, ToolCallList]] = None
        successful_idx: Optional[int] = None

        for idx in provider_indices:
            provider = self._providers[idx]
            try:
                stream_gen = provider.async_stream(**kwargs_stream)
                successful_idx = idx
                break
            except LLMError as e:
                last_err = e
                self._logger.warning("Provider %s failed to start stream: %s", provider.__class__.__name__, e)
                continue
            except Exception as e:
                self._logger.warning("Provider %s failed to start stream: %s", provider.__class__.__name__, e)
                raise

        if stream_gen is None:
            raise RuntimeError(f"All LLM providers failed to stream. Last error: {last_err}")

        if successful_idx is not None:
            self._primary_provider_idx = successful_idx

        async for chunk in stream_gen:
            try:
                # Forward recognised response types as-is. Allow raw strings (deltas) to flow through, too.
                if isinstance(chunk, LLMResponse):
                    if chunk.type == "llm_delta":
                        yield ChatChunk(role="assistant", 
                                        content=chunk.text, 
                                        model=chunk.model_id, 
                                        usage=None)
                    elif chunk.type == "llm_response":
                        yield AssistantMessage(role="assistant", 
                                               content=chunk.text,  # type: ignore
                                               model=chunk.model_id, 
                                               usage=LLMUsage(prompt_tokens=chunk.usage.prompt_tokens,
                                                              completion_tokens=chunk.usage.completion_tokens,
                                                              total_tokens=chunk.usage.total_tokens))
                elif isinstance(chunk, ToolCallList):
                    yield ToolCallRequestList(tool_calls=[ToolCallRequest(role="tool_call",
                                                                         id=tool_call.id,
                                                                         type="function_call",
                                                                         name=tool_call.function_name,
                                                                         arguments=tool_call.arguments) for tool_call in chunk.tool_calls],
                                              usage=LLMUsage(prompt_tokens=chunk.usage.prompt_tokens,
                                                             completion_tokens=chunk.usage.completion_tokens,
                                                             total_tokens=chunk.usage.total_tokens))
                else:
                    self._logger.warning("Unknown chunk type from provider stream: %s", type(chunk))
                    continue

            except Exception as e:
                self._logger.warning("Provider %s failed to stream: %s", provider.__class__.__name__, e)
                raise
