"""Parameter sources resolve against request context without blocking the event loop."""

import asyncio
from contextvars import ContextVar
from threading import get_ident

import pytest
from pydantic import BaseModel

from seqoria_agent.client.parameters import ParamContext, ParameterRegistry


@pytest.mark.asyncio
async def test_constant_source():
    reg = ParameterRegistry({"foo": 42})
    ctx: ParamContext = {}
    assert await reg.resolve("foo", context=ctx) == 42


@pytest.mark.asyncio
async def test_sync_callable():

    def provide_tenant(tenant_id: str | None = None):
        return f"tenant-{tenant_id}" if tenant_id else "no-tenant"

    reg = ParameterRegistry({"tenant": provide_tenant})
    ctx: ParamContext = {"tenant_id": "acme"}
    assert await reg.resolve("tenant", context=ctx) == "tenant-acme"


@pytest.mark.asyncio
async def test_async_callable():

    async def provide_user(user_id: str | None = None):
        await asyncio.sleep(0)
        return f"user-{user_id}" if user_id else "anon"

    reg = ParameterRegistry({"user": provide_user})
    ctx: ParamContext = {"user_id": "u-123"}
    assert await reg.resolve("user", context=ctx) == "user-u-123"


@pytest.mark.asyncio
async def test_missing_param_returns_none():
    reg = ParameterRegistry({})
    ctx: ParamContext = {}
    assert await reg.resolve("unknown", context=ctx) is None


@pytest.mark.asyncio
async def test_kwarg_filtering():

    def needs_history(history):
        return len(history)

    reg = ParameterRegistry({"h_len": needs_history})
    ctx: ParamContext = {"history": [{"role": "user", "content": "hi"}], "tenant_id": "skip"}
    assert await reg.resolve("h_len", context=ctx) == 1


class Address(BaseModel):
    street: str
    city: str
    country: str


@pytest.mark.asyncio
async def test_basemodel_key_resolution():

    def provide_address():
        return Address(street="123", city="NYC", country="US")

    reg = ParameterRegistry({Address: provide_address})
    ctx: ParamContext = {}
    result = await reg.resolve("address", context=ctx)
    assert isinstance(result, Address)
    assert result.street == "123"


@pytest.mark.asyncio
async def test_callable_exception_swallowed(caplog):

    def bad_callable():
        raise RuntimeError("boom")

    reg = ParameterRegistry({"bad": bad_callable})
    ctx: ParamContext = {}
    with caplog.at_level("WARNING"):
        assert await reg.resolve("bad", context=ctx) is None
        assert any(("failed to resolve" in rec.message for rec in caplog.records))


@pytest.mark.asyncio
async def test_context_override_takes_precedence():
    reg = ParameterRegistry({"foo": lambda: "registry"})
    ctx: ParamContext = {"foo": "context"}
    assert await reg.resolve("foo", context=ctx) == "context"


@pytest.mark.asyncio
async def test_request_context_precedes_registry_without_mutating_defaults():
    registry = ParameterRegistry({"user_id": "default"})
    assert await registry.resolve("user_id", context={"user_id": "request"}) == "request"
    assert await registry.resolve("user_id", context={}) == "default"


@pytest.mark.asyncio
async def test_sync_parameter_providers_run_off_loop_with_request_context():
    request_identity = ContextVar("identity", default="missing")
    token = request_identity.set("authenticated-tenant")
    event_loop_thread = get_ident()
    calls = []

    def provider(*args):
        calls.append((get_ident(), request_identity.get()))
        return request_identity.get()

    try:
        registry = ParameterRegistry({"tenant": provider})
        result = await registry.resolve("tenant", context={})
        assert result == "authenticated-tenant"
        assert calls == [(calls[0][0], "authenticated-tenant")]
        assert calls[0][0] != event_loop_thread
    finally:
        request_identity.reset(token)
