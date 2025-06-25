import pytest
from types import SimpleNamespace

from seqoria_agent.server.xmcp_core import AuthBridge
from seqoria_agent.auth.backends import BaseAuthBackend, NoAuthBackend
from seqoria_agent.models.errors import XMCPError


class DummyBackend(BaseAuthBackend):
    """Backend that always returns a principal object and counts calls."""

    def __init__(self):
        self.calls = 0

    async def authenticate(self, request):  # noqa: D401
        self.calls += 1
        return {"id": "alice"}


class SilentBackend(BaseAuthBackend):
    """Backend that returns *None* while *requiring* authentication."""

    async def authenticate(self, request):  # noqa: D401
        return None


@pytest.mark.asyncio
async def test_noauth_backend_allows_none_principal():
    bridge = AuthBridge(NoAuthBackend())
    req = SimpleNamespace(state=SimpleNamespace())

    # NoAuthBackend returns None → should not raise
    await bridge.authenticate(req)
    # ensure_principal_or_error must *not* raise for NoAuthBackend
    bridge.ensure_principal_or_error("whoami")
    assert getattr(req.state, "principal", "missing") is None


@pytest.mark.asyncio
async def test_principal_stored_once_for_dummy_backend():
    backend = DummyBackend()
    bridge = AuthBridge(backend)
    req = SimpleNamespace(state=SimpleNamespace())

    # First invocation performs authentication
    await bridge.authenticate(req)
    # Second invocation should be a no-op (principal already present)
    await bridge.authenticate(req)

    assert backend.calls == 1  # exactly one backend call
    assert getattr(req.state, "principal", None) == {"id": "alice"}
    # principal present → ensure helper does not raise
    bridge.ensure_principal_or_error("tool_x")


@pytest.mark.asyncio
async def test_backend_requires_auth_but_returns_none():
    bridge = AuthBridge(SilentBackend())
    req = SimpleNamespace(state=SimpleNamespace())

    await bridge.authenticate(req)
    with pytest.raises(XMCPError) as exc:
        bridge.ensure_principal_or_error("secure_tool")
    assert exc.value.code == "UNAUTHENTICATED" 