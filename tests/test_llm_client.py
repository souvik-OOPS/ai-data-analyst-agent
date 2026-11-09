"""ClaudeClient against the real Anthropic SDK with a mocked HTTP transport (no network, no API key)."""

from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from app.agent.llm import FALLBACK_BETA, ClaudeClient, LLMUnavailableError
from app.config import Settings

OK_MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def make_client(handler, fallback: bool = True) -> ClaudeClient:
    settings = Settings(_env_file=None, anthropic_api_key="test-key", llm_refusal_fallback=fallback,
                        llm_max_retries=0)
    sdk = anthropic.Anthropic(api_key="test-key", max_retries=0,
                              http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return ClaudeClient(settings, client=sdk)


def call(client: ClaudeClient):
    return client.create(system="s", messages=[{"role": "user", "content": "hi"}],
                         tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}])


def error(status: int, kind: str, message: str = "boom"):
    return lambda request: httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})


def test_request_shape():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, json=OK_MESSAGE)

    response = call(make_client(handler))
    assert response.content[0].text == "hi"
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default" and seen["beta"] == FALLBACK_BETA
    assert body["tools"][0]["name"] == "t" and "thinking" not in body and "temperature" not in body


@pytest.mark.parametrize("status,kind,expected", [
    (401, "authentication_error", "auth"),
    (403, "permission_error", "permission"),
    (404, "not_found_error", "model"),
    (429, "rate_limit_error", "rate_limit"),
    (400, "invalid_request_error", "bad_request"),
    (500, "api_error", "server"),
    (529, "overloaded_error", "server"),
])
def test_http_errors_are_classified(status, kind, expected):
    with pytest.raises(LLMUnavailableError) as exc:
        call(make_client(error(status, kind), fallback=False))
    assert exc.value.kind == expected


def test_timeout_and_connection_errors():
    def timeout(request):
        raise httpx2.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailableError) as exc:
        call(make_client(timeout))
    assert exc.value.kind == "timeout"
    with pytest.raises(LLMUnavailableError) as exc:
        call(make_client(refused))
    assert exc.value.kind == "connection"


def test_refusal_is_treated_as_unavailable():
    refusal = {**OK_MESSAGE, "content": [], "stop_reason": "refusal"}
    with pytest.raises(LLMUnavailableError) as exc:
        call(make_client(lambda request: httpx2.Response(200, json=refusal)))
    assert exc.value.kind == "refusal"


def test_unavailable_fallback_beta_is_disabled_and_request_retried():
    requests = []

    def handler(request):
        requests.append(request.headers.get("anthropic-beta"))
        if request.headers.get("anthropic-beta"):
            return error(400, "invalid_request_error",
                         f"Unexpected value(s) `{FALLBACK_BETA}` for the `anthropic-beta` header.")(request)
        return httpx2.Response(200, json=OK_MESSAGE)

    client = make_client(handler)
    assert call(client).content[0].text == "hi"
    assert call(client).content[0].text == "hi"
    assert requests == [FALLBACK_BETA, None, None]  # disabled after the first rejection
