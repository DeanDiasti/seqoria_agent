import pytest

from seqoria_agent.client.ai_client.history import build_system_prompt, summarise_history
from seqoria_agent.client.ai_client.callbacks import on_display, on_error, on_metadata


def test_build_system_prompt_extra():
    extra = "Please respond in a formal tone."
    prompt = build_system_prompt(extra_instructions=extra)
    assert extra in prompt
    assert "__clarify__" in prompt  # core prompt still present


def test_summarise_history():
    history = [
        {"role": "user", "content": f"msg{i}"} for i in range(10)
    ]
    summary = summarise_history(history)
    # Should contain the last 6 messages (msg4 .. msg9)
    for i in range(4, 10):
        assert f"msg{i}" in summary
    assert "msg3" not in summary  # earlier message excluded


@pytest.mark.parametrize("wrapper, payload", [
    (on_display, {"foo": "bar"}),
    (on_error, {"err": "oops"}),
    (on_metadata, {"meta": 123}),
])
def test_callback_wrappers(wrapper, payload):
    called = {}

    def good_cb(obj, hist):  # pylint: disable=unused-argument
        called["ok"] = True

    wrapper(payload, history=[], cb=good_cb)
    assert called.get("ok") is True

    # Callback raises – wrapper should swallow and not propagate
    def bad_cb(obj, hist):  # pylint: disable=unused-argument
        raise RuntimeError("boom")

    # Should not raise
    wrapper(payload, history=[], cb=bad_cb) 