"""Gemini backend: message conversion, response parsing, error mapping and model failover.

Uses real google-genai types with a fake ``models.generate_content`` (no network, no quota).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app import analytics
from app.agent.agent import AnalystAgent
from app.agent.gemini import (
    GeminiClient,
    RawGeminiContent,
    TextBlock,
    ToolUseBlock,
    from_gemini_response,
    to_gemini_contents,
)
from app.agent.llm import LLMUnavailableError
from app.config import Settings
from app.formatting import money


def g_response(*parts, finish="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)), finish_reason=finish)])


def call_part(name, args, id="call_1"):
    return types.Part(function_call=types.FunctionCall(id=id, name=name, args=args))


def quota_error(delay="30s"):
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED",
                                              "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                           "retryDelay": delay}]}})


def api_error(cls, code, message="boom"):
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes, fallbacks="m2"):
    settings = Settings(_env_file=None, gemini_api_key="test", gemini_model="m1", gemini_fallback_models=fallbacks)
    fake = SimpleNamespace(models=FakeModels(outcomes))
    return GeminiClient(settings, client=fake), fake.models


# --------------------------------------------------------------------------- conversion


def test_messages_are_converted_to_gemini_contents():
    messages = [
        {"role": "user", "content": "Top products?"},
        {"role": "assistant", "content": [TextBlock("Checking."),
                                          ToolUseBlock(id="local-1", name="get_top_products", input={"limit": 5})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "local-1",
                                      "content": json.dumps({"products": [1]}), "is_error": False}]},
        {"role": "user", "content": "Please verify."},
    ]
    contents = to_gemini_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user", "user"]
    call = contents[1].parts[1].function_call
    assert (call.name, call.args, call.id) == ("get_top_products", {"limit": 5}, None)  # local ids are not sent
    response = contents[2].parts[0].function_response
    assert response.name == "get_top_products" and response.response == {"result": {"products": [1]}}


def test_error_results_and_raw_turns():
    raw = types.Content(role="model", parts=[call_part("run_sql", {"query": "x"}, id="c9")])
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [ToolUseBlock("c9", "run_sql", {"query": "x"}), RawGeminiContent(raw, "m1")]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c9", "content": '{"error": "bad"}',
                                      "is_error": True}]},
    ]
    contents = to_gemini_contents(messages)
    assert contents[1] is raw  # replayed unchanged, keeping thought signatures
    fr = contents[2].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("c9", "run_sql", {"error": {"error": "bad"}})


def test_response_parsing():
    r = from_gemini_response(g_response(types.Part(text="thinking...", thought=True),
                                        call_part("get_sales_metrics", {"start_date": "2026-10-01"})), "m1")
    assert r.stop_reason == "tool_use"
    assert [b.type for b in r.content] == ["tool_use", "gemini_raw"]  # thought text is not shown to users
    assert r.content[0].input == {"start_date": "2026-10-01"} and r.content[1].model == "m1"
    assert from_gemini_response(g_response(types.Part(text="Done."))).stop_reason == "end_turn"
    assert from_gemini_response(g_response(types.Part(text="Partial"), finish="MAX_TOKENS")).stop_reason == "max_tokens"


def test_blocked_responses_are_refusals():
    with pytest.raises(LLMUnavailableError) as exc:
        from_gemini_response(g_response(types.Part(text="x"), finish="SAFETY"))
    assert exc.value.kind == "refusal"
    blocked = types.GenerateContentResponse(candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason="SAFETY"))
    with pytest.raises(LLMUnavailableError) as exc:
        from_gemini_response(blocked)
    assert exc.value.kind == "refusal"


# --------------------------------------------------------------------------- requests and errors


def test_request_configuration():
    client, fake = make_client([g_response(types.Part(text="ok"))])
    client.create(system="be precise", messages=[{"role": "user", "content": "hi"}],
                  tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
                  output_format={"type": "json_schema", "schema": {"type": "object"}})
    config = fake.calls[0].config
    assert fake.calls[0].model == "m1"
    assert config.system_instruction == "be precise"
    assert config.automatic_function_calling.disable is True  # the agent runs tools itself
    assert config.tools[0].function_declarations[0].name == "t"
    assert config.response_mime_type == "application/json"


@pytest.mark.parametrize("error,kind,retryable", [
    (api_error(errors.ClientError, 401, "API key not valid"), "auth", False),
    (api_error(errors.ClientError, 400), "bad_request", False),
    (api_error(errors.ClientError, 404), "model", True),
    (quota_error(), "rate_limit", True),
    (api_error(errors.ServerError, 503), "overloaded", True),
    (api_error(errors.ServerError, 500), "server", False),
    (httpx.ReadTimeout("slow"), "timeout", False),
    (httpx.ConnectError("refused"), "connection", False),
])
def test_errors_are_classified(error, kind, retryable):
    client, _ = make_client([error])
    with pytest.raises(LLMUnavailableError) as exc:
        client.create(system="s", messages=[{"role": "user", "content": "hi"}])
    assert (exc.value.kind, exc.value.retryable) == (kind, retryable)


def test_quota_error_moves_to_next_model():
    client, fake = make_client([quota_error("120s"), g_response(types.Part(text="ok")),
                                api_error(errors.ServerError, 503)])
    with pytest.raises(LLMUnavailableError):
        client.create(system="s", messages=[{"role": "user", "content": "hi"}])
    assert client.model == "m2" and client.has_available_model()
    client.create(system="s", messages=[{"role": "user", "content": "hi"}])
    assert fake.calls[1].model == "m2"
    with pytest.raises(LLMUnavailableError):
        client.create(system="s", messages=[{"role": "user", "content": "hi"}])
    assert not client.has_available_model()


def test_conversation_stays_on_its_model():
    client, fake = make_client([g_response(types.Part(text="ok"))])
    client._cool_down("m1", 60)
    raw = RawGeminiContent(types.Content(role="model", parts=[types.Part(text="earlier")]), "m1")
    client.create(system="s", messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": [raw]},
                                        {"role": "user", "content": "follow-up"}])
    assert fake.calls[0].model == "m1"


def test_agent_fails_over_to_next_model(registry, full_info, full_db):
    top = analytics.product_performance(full_db, limit=1)["products"][0]
    client, fake = make_client([
        quota_error(),
        g_response(call_part("get_top_products", {"limit": 5})),
        g_response(types.Part(text=f"{top['product_name']} leads with {money(top['revenue'])}.")),
    ])
    answer = AnalystAgent(registry, full_info, client).ask("What were the top 5 products by revenue?")
    assert (answer.engine, answer.model, answer.status) == ("llm", "m2", "answered")
    assert answer.grounding.grounded and [c.tool for c in answer.tool_calls] == ["get_top_products"]
    assert "fallback model m2" in answer.warnings[-1]
    assert [c.model for c in fake.calls] == ["m1", "m2", "m2"]


def test_insight_writer_fails_over_to_next_model(full_db, full_info):
    from app.agent.insight_writer import write_insights

    base = analytics.compute_insights(full_db, full_info)[:1]
    rewritten = {"insights": [{"id": base[0]["id"], "title": "October drop", "text": base[0]["text"]}]}
    client, fake = make_client([api_error(errors.ServerError, 503), g_response(types.Part(text=json.dumps(rewritten)))])
    items, warnings = write_insights(client, base)
    assert items[0]["source"] == "llm" and items[0]["title"] == "October drop" and warnings == []
    assert [c.model for c in fake.calls] == ["m1", "m2"]
    assert fake.calls[1].config.response_mime_type == "application/json"


def test_answer_reports_the_model_that_actually_answered(registry, full_info, full_db):
    top = analytics.product_performance(full_db, limit=1)["products"][0]
    client = None

    def tool_call_then_primary_recovers():
        client._cooldown_until.clear()  # m1's cool-down expires while the conversation runs on m2
        return g_response(call_part("get_top_products", {"limit": 5}))

    client, fake = make_client([
        api_error(errors.ServerError, 503),
        tool_call_then_primary_recovers,
        g_response(types.Part(text=f"{top['product_name']} leads with {money(top['revenue'])}.")),
    ])
    answer = AnalystAgent(registry, full_info, client).ask("What were the top 5 products by revenue?")
    assert [c.model for c in fake.calls] == ["m1", "m2", "m2"]  # the conversation stays on m2
    assert answer.model == "m2"  # reported model is the one that answered, not the current default
