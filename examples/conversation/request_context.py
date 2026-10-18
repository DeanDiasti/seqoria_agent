"""Share a client while keeping each request's identity and history separate."""

import asyncio

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.models.chunk import AssistantMessage, ToolCallResult
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, ToolCallList

toolbox = LocalToolbox()


@toolbox.tool(server_populated=["user_id", "tenant_id"])
def account_label(user_id: str, tenant_id: str) -> str:
    """Return the authenticated account label."""
    return f"{tenant_id}/{user_id}"


def resolve_user(authenticated_user_id: str) -> str:
    """An application resolver receives trusted middleware context."""
    return authenticated_user_id


class ScriptedProvider(AsyncBaseProvider):
    """A stateless stand-in for a shared async model provider."""

    async def async_generate(self, chat_history, **kwargs):
        last_message = chat_history[-1]
        if isinstance(last_message, ToolCallResult):
            return LLMResponse(model_id="offline", type="llm_response", text=last_message.result)
        return ToolCallList(
            model_id="offline",
            tool_calls=[
                ToolCall(
                    id="account-1",
                    type="function_call",
                    function_name="account_label",
                    arguments={},
                )
            ],
            usage=None,
        )

    async def async_stream(self, chat_history, **kwargs):
        yield await self.async_generate(chat_history=chat_history, **kwargs)


async def serve_request(client: AsyncAIClient, *, user_id: str, tenant_id: str) -> tuple[str, int]:
    # A web application loads and persists this list under its authenticated conversation ID.
    # Never share one mutable history list between concurrent conversations.
    history = []
    context = {"authenticated_user_id": user_id, "tenant_id": tenant_id}
    async for chunk in client.chat(
        "Which account am I using?", history=history, context=context, max_turns=3
    ):
        if isinstance(chunk, AssistantMessage):
            return str(chunk.content), len(history)
    raise RuntimeError("No assistant reply")


async def main() -> None:
    parameters = ParameterRegistry({"user_id": resolve_user})
    async with (
        ScriptedProvider() as provider,
        AsyncAIClient(
            providers=[provider],
            local_toolbox=toolbox,
            param_registry=parameters,
            max_concurrency=2,
        ) as client,
    ):
        replies = await asyncio.gather(
            serve_request(client, user_id="alice", tenant_id="tenant-a"),
            serve_request(client, user_id="bob", tenant_id="tenant-b"),
        )
        for reply, message_count in replies:
            print(f"{reply} ({message_count} messages in its own history)")


if __name__ == "__main__":
    asyncio.run(main())
