from __future__ import annotations

"""Provider interaction utilities for *seqoria_agent*'s new AIClient implementation.

This module is introduced in **Phase 4 – LLMInterface extraction** (see
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

from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.schemas import LLMResponse

__all__: list[str] = [
    "LLMInterface",
]


class LLMInterface:  # noqa: D101 – module-level docstring provides context
    # LOC gate: keep implementation short (< 250 lines)

    def __init__(self, providers: List[BaseProvider], *, logger: Optional[logging.Logger] = None) -> None:
        if not providers:
            raise ValueError("'providers' must contain at least one BaseProvider instance.")
        self._providers: List[BaseProvider] = providers
        if logger is not None:
            self._logger = logger
        else:
            self._logger = (
                structlog.get_logger("seqoria_agent.llm_interface") if structlog is not None else logging.getLogger("seqoria_agent.llm_interface")
            )

    # ------------------------------------------------------------------
    # Public API – synchronous generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        history: List[dict],
        *,
        tools: List[dict] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Generate a single response using the first provider that succeeds.

        Parameters
        ----------
        history
            The chat history in OpenAI format (mutated by the caller).
        tools, tool_choice
            Optional tool specification forwarded to the provider.

        Returns
        -------
        LLMResponse
            The model's answer.

        Raises
        ------
        RuntimeError
            If *all* providers fail with :class:`LLMError`.
        """
        last_err: Optional[Exception] = None

        for provider in self._providers:
            try:
                kwargs_gen: dict[str, Any] = {"chat_history": history}
                if tools is not None:
                    kwargs_gen["tools"] = tools
                    kwargs_gen["tool_choice"] = tool_choice

                raw_resp = await provider.async_generate(**kwargs_gen)  # type: ignore[arg-type]

                # Coerce into LLMResponse for uniform downstream handling -----------------
                if isinstance(raw_resp, LLMResponse):
                    response = raw_resp
                else:
                    try:
                        response = LLMResponse.model_validate(raw_resp)  # type: ignore[arg-type]
                    except Exception:
                        # Minimal best-effort fallback – map common attributes if present
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
            except Exception:
                # Do not swallow unexpected exceptions
                raise

        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

    # ------------------------------------------------------------------
    # Public API – streaming generation
    # ------------------------------------------------------------------

    async def stream(
        self,
        history: List[dict],
        *,
        tools: List[dict] | None = None,
        tool_choice: str = "auto",
    ) -> AsyncGenerator[Union[str, LLMResponse], None]:
        """Yield incremental deltas followed by the final :class:`LLMResponse`.

        The method attempts each provider in order until a stream generator is
        obtained.  It proxies every delta (``str``) downstream immediately and
        finally yields the *full* ``LLMResponse`` once the provider signals
        completion.
        """
        # ------------------------------------------------------------------
        # 1) Select provider that successfully starts a stream
        # ------------------------------------------------------------------
        stream_gen: Optional[AsyncGenerator[Union[str, LLMResponse], None]] = None
        last_err: Optional[Exception] = None

        for provider in self._providers:
            try:
                kwargs_stream = {"chat_history": history}
                if tools is not None:
                    kwargs_stream["tools"] = tools
                    kwargs_stream["tool_choice"] = tool_choice

                tmp = provider.async_stream(**kwargs_stream)  # type: ignore[arg-type]

                # Guard against both async and sync implementations
                stream_gen = await tmp if asyncio.iscoroutine(obj=tmp) else tmp

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

        # ------------------------------------------------------------------
        # 2) Proxy the provider stream while tracking the final answer
        # ------------------------------------------------------------------
        aggregated_text: list[str] = []
        final_response: Optional[LLMResponse] = None

        async for chunk in stream_gen:
            if isinstance(chunk, str):
                aggregated_text.append(chunk)
                yield chunk
            else:
                # Accept both LLMResponse and plain dict for provider flexibility
                if isinstance(chunk, LLMResponse):
                    final_response = chunk
                elif isinstance(chunk, dict):
                    try:
                        final_response = LLMResponse.model_validate(chunk)
                    except Exception:
                        final_response = LLMResponse(
                            model_id=chunk.get("model_id", "unknown"),
                            text=chunk.get("text"),
                            tool_calls=chunk.get("tool_calls"),
                            usage=chunk.get("usage"),
                            stop_reason=chunk.get("stop_reason"),
                        )
                else:
                    self._logger.warning("Unknown chunk type from provider stream", type=str(type(chunk)))
            # Yield non-string chunks only *after* they are processed (see below)

        # Some providers may finish without emitting a distinct LLMResponse
        if final_response is None:
            final_response = LLMResponse(
                model_id="unknown",
                text="".join(aggregated_text),
            )

        yield final_response 