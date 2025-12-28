import pytest

from seqoria_agent.llm_providers import errors as err


@pytest.mark.parametrize(
    "error_cls, kwargs, expected_fragment",
    [
        (err.EnvironmentError, {"missing_vars": ["FOO_VAR"]}, "FOO_VAR"),
        (err.ConnectionError, {"provider": "Dummy"}, "Dummy"),
        (err.ModelNotFoundError, {"model_id": "foo", "provider": "Dummy"}, "foo"),
        (err.ContextLengthExceededError, {"model_id": "m", "tokens": 42, "max_tokens": 20}, "42"),
        (err.ContentFilterError, {"provider": "Dummy", "filter_reason": "policy"}, "policy"),
        (err.MaxRetriesExceededError, {"operation": "op", "max_retries": 2}, "op"),
        (err.InvalidRequestError, {"message": "bad"}, "bad"),
        (err.QuotaExceededError, {"provider": "Dummy", "limit_type": "rate"}, "rate"),
        (err.ServerError, {"provider": "Dummy", "status_code": 500}, "500"),
    ],
)
def test_error_message_contains_context(error_cls, kwargs, expected_fragment):
    exc = error_cls(**kwargs)  # type: ignore[arg-type]
    assert expected_fragment in str(exc) 