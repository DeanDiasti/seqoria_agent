from __future__ import annotations

"""Provider fallback and conversion to the client message protocol."""

from typing import Any, AsyncGenerator, List, Optional, Union

import logging
from contextlib import suppress

from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ToolCallRequest,
    ChatChunk,
    ToolCallRequestList,
    LLMUsage,
)

__all__: list[str] = ["LLMInterface"]


class LLMInterface:  # noqa: D101 – module-level docstring provides context
    def __init__(
        self, providers: List[AsyncBaseProvider], *, logger: Optional[logging.Logger] = None
    ) -> None:
        if not providers:
            raise ValueError("'providers' must contain at least one AsyncBaseProvider instance.")
        self._providers: List[AsyncBaseProvider] = list(providers)
        # Cache index of provider that succeeded last time – optimistic fast-path
        self._primary_provider_idx: int = 0
        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(name="seqoria_agent.llm_interface")

    # ------------------------------------------------------------------
    # Public API – synchronous generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        history: List[Any],
        *,
        tools: List[dict] | None = None,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> Union[AssistantMessage, ToolCallRequestList]:
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
        kwargs_gen_base: dict[str, Any] = {"chat_history": history}
        kwargs_gen_base["tool_choice"] = (tool_choice or "auto") if tools else None
        kwargs_gen_base["model"] = model

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
                kwargs_gen = dict(kwargs_gen_base)
                kwargs_gen["tools"] = tools if tools is not None else None
                raw_resp = await provider.async_generate(**kwargs_gen)
                successful_idx = idx
                break  # success
            except LLMError as e:
                last_err = e
                self._logger.warning(
                    "Provider %s failed to generate: %s", provider.__class__.__name__, e
                )
                continue
            except Exception as e:
                # Non-LLM errors are not considered retryable – re-raise.
                self._logger.warning(
                    "Provider %s failed to generate: %s", provider.__class__.__name__, e
                )
                raise

        if raw_resp is None:
            raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

        # Update fast-path cache on success
        if successful_idx is not None:
            self._primary_provider_idx = successful_idx

        if isinstance(raw_resp, LLMResponse):
            usage = raw_resp.usage or LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return AssistantMessage(
                role="assistant",
                content=raw_resp.text or "",
                thinking=raw_resp.thinking,
                model=raw_resp.model_id,
                usage=LLMUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

        if isinstance(raw_resp, ToolCallList):
            usage = raw_resp.usage or LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return ToolCallRequestList(
                tool_calls=[
                    ToolCallRequest(
                        role="tool_call",
                        id=tool_call.id,
                        type="function_call",
                        name=tool_call.function_name,
                        arguments=tool_call.arguments,
                    )
                    for tool_call in raw_resp.tool_calls
                ],
                usage=LLMUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

        raise TypeError(f"Unexpected provider response: {type(raw_resp)}")

    # ------------------------------------------------------------------
    # Public API – streaming generation
    # ------------------------------------------------------------------

    async def stream(
        self,
        history: List[Any],
        *,
        tools: List[dict] | None = None,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> AsyncGenerator[Union[ChatChunk, AssistantMessage, ToolCallRequestList], None]:
        """Yield incremental deltas followed by the final :class:`LLMResponse`.

        The method attempts each provider in order until a stream generator is
        obtained.  It proxies every delta (``str``) downstream immediately and
        finally yields the *full* ``LLMResponse`` once the provider signals
        completion.
        """

        kwargs_stream_base: dict[str, Any] = {"chat_history": history}
        kwargs_stream_base["tool_choice"] = (tool_choice or "auto") if tools else None
        kwargs_stream_base["model"] = model

        last_err = None
        indices = [self._primary_provider_idx] + [
            idx for idx in range(len(self._providers)) if idx != self._primary_provider_idx
        ]
        for idx in indices:
            provider = self._providers[idx]
            emitted = False
            stream_gen = None
            try:
                stream_gen = provider.async_stream(**kwargs_stream_base, tools=tools)
                async for chunk in stream_gen:
                    if isinstance(chunk, LLMResponse):
                        if chunk.type == "llm_delta":
                            # Some providers (e.g. Ollama with think=True) may emit "thinking" deltas
                            # with text=None. Do not emit invalid ChatChunk(content=None).
                            if not chunk.text and not chunk.thinking:
                                continue
                            emitted = True
                            self._primary_provider_idx = idx
                            if chunk.text is None:
                                thinking_delta = chunk.thinking
                                # Forward thinking deltas via the explicit field.
                                yield ChatChunk(
                                    role="assistant",
                                    content="",
                                    thinking=thinking_delta,
                                    model=chunk.model_id,
                                    usage=None,
                                )
                                continue
                            yield ChatChunk(
                                role="assistant",
                                content=chunk.text,
                                thinking=chunk.thinking,
                                model=chunk.model_id,
                                usage=None,
                            )
                        elif chunk.type == "llm_response":
                            emitted = True
                            self._primary_provider_idx = idx
                            usage = chunk.usage or LLMUsage(
                                prompt_tokens=0, completion_tokens=0, total_tokens=0
                            )
                            yield AssistantMessage(
                                role="assistant",
                                content=chunk.text or "",
                                thinking=chunk.thinking,
                                model=chunk.model_id,
                                usage=LLMUsage(
                                    prompt_tokens=usage.prompt_tokens,
                                    completion_tokens=usage.completion_tokens,
                                    total_tokens=usage.total_tokens,
                                ),
                            )
                    elif isinstance(chunk, ToolCallList):
                        emitted = True
                        self._primary_provider_idx = idx
                        usage = chunk.usage or LLMUsage(
                            prompt_tokens=0, completion_tokens=0, total_tokens=0
                        )
                        yield ToolCallRequestList(
                            tool_calls=[
                                ToolCallRequest(
                                    role="tool_call",
                                    id=tool_call.id,
                                    type="function_call",
                                    name=tool_call.function_name,
                                    arguments=tool_call.arguments,
                                )
                                for tool_call in chunk.tool_calls
                            ],
                            usage=LLMUsage(
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                total_tokens=usage.total_tokens,
                            ),
                        )
                    else:
                        raise TypeError(f"Unknown chunk type from provider stream: {type(chunk)}")

                if not emitted:
                    raise LLMError(message="Provider stream ended without a response.")
                return
            except LLMError as exc:
                if emitted:
                    raise
                last_err = exc
                self._logger.warning(
                    "Provider %s failed to start stream: %s", provider.__class__.__name__, exc
                )
            finally:
                if stream_gen is not None:
                    with suppress(Exception):
                        await stream_gen.aclose()
        raise RuntimeError(f"All LLM providers failed to stream. Last error: {last_err}")
