"""Agent loop behaviour with a scripted fake LLM: tool execution, validation, grounding and fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import analytics
from app.agent.agent import AnalystAgent
from app.agent.llm import LLMUnavailableError
from app.agent.prompts import CORRECTION_NO_TOOLS
from app.errors import InvalidInputError
from app.formatting import money


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool(name: str, args: dict, id: str = "toolu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=args)


def reply(*blocks, stop: str | None = None):
    has_tool = any(b.type == "tool_use" for b in blocks)
    return SimpleNamespace(content=list(blocks), stop_reason=stop or ("tool_use" if has_tool else "end_turn"),
                           usage=None)


class FakeLLM:
    """Returns scripted responses in order and records the messages it was sent."""

    model = "fake-model"

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def create(self, *, system, messages, tools=None, output_format=None, effort=None):
        self.calls.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def top_product(full_db):
    return analytics.product_performance(full_db, limit=1)["products"][0]


def make_agent(registry, info, script, **kwargs):
    llm = FakeLLM(script)
    return AnalystAgent(registry, info, llm, **kwargs), llm


def test_tool_loop_produces_grounded_answer(registry, full_info, top_product):
    agent, llm = make_agent(registry, full_info, [
        reply(tool("get_top_products", {"limit": 5})),
        reply(text(f"The top product is {top_product['product_name']} with {money(top_product['revenue'])}.")),
    ])
    a = agent.ask("What were the top 5 products by revenue?")
    assert (a.engine, a.status, a.model) == ("llm", "answered", "fake-model")
    assert a.grounding.grounded and a.grounding.verified_numbers == 1
    assert [c.tool for c in a.tool_calls] == ["get_top_products"]
    result_block = llm.calls[1][-1]["content"][0]
    assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "toolu_1"
    assert result_block["is_error"] is False and top_product["product_name"] in result_block["content"]


def test_invented_figure_is_corrected_by_the_model(registry, full_info, top_product):
    agent, llm = make_agent(registry, full_info, [
        reply(tool("get_top_products", {"limit": 5})),
        reply(text("The top product made $999,999.")),
        reply(text(f"The top product made {money(top_product['revenue'])}.")),
    ])
    a = agent.ask("What were the top 5 products by revenue?")
    assert a.engine == "llm" and a.grounding.grounded
    assert "999,999" in llm.calls[2][-1]["content"]  # the correction named the bad figure


def test_persistently_invented_figures_fall_back_to_verified_answer(registry, full_info, top_product):
    agent, _ = make_agent(registry, full_info, [
        reply(tool("get_top_products", {"limit": 5})),
        reply(text("The top product made $999,999.")),
        reply(text("Sorry - it made $888,888.")),
    ])
    a = agent.ask("What were the top 5 products by revenue?")
    assert a.engine == "rule_based" and a.grounding.grounded
    assert "could not be verified" in a.warnings[0] and "888,888" in a.warnings[0]
    assert top_product["product_name"] in a.answer
    assert "999,999" not in a.answer and "888,888" not in a.answer


def test_invalid_tool_arguments_are_returned_to_the_model(registry, full_info, top_product):
    agent, llm = make_agent(registry, full_info, [
        reply(tool("get_top_products", {"limit": 500})),
        reply(tool("get_top_products", {"limit": 5}, id="toolu_2")),
        reply(text(f"{top_product['product_name']} leads.")),
    ])
    a = agent.ask("What were the top 5 products by revenue?")
    error_block = llm.calls[1][-1]["content"][0]
    assert error_block["is_error"] is True and "limit" in error_block["content"]
    assert [c.ok for c in a.tool_calls] == [False, True]
    assert a.status == "answered"


def test_unknown_tool_is_an_error_not_a_crash(registry, full_info):
    agent, llm = make_agent(registry, full_info, [
        reply(tool("drop_database", {})),
        reply(text("I can't do that.")),
    ])
    agent.ask("Which category generated the most revenue?")
    block = llm.calls[1][-1]["content"][0]
    assert block["is_error"] is True and "Unknown tool" in block["content"]


def test_unsupported_question_is_flagged(registry, full_info):
    agent, _ = make_agent(registry, full_info, [
        reply(tool("flag_unsupported", {"reason": "There is no cost data, so margins cannot be computed.",
                                        "suggestion": "What was revenue by category?"})),
    ])
    a = agent.ask("What is our gross margin?")
    assert a.status == "unsupported" and "no cost data" in a.answer and "revenue by category" in a.answer


def test_llm_api_failure_falls_back_to_rule_engine(registry, full_info):
    agent, _ = make_agent(registry, full_info, [LLMUnavailableError("rate_limit", "the API rate limit was exceeded")])
    a = agent.ask("Which category generated the most revenue?")
    assert a.engine == "rule_based" and a.status == "answered"
    assert "rate limit" in a.warnings[0]
    assert "Electronics" in a.answer


def test_step_limit_stops_runaway_loops(registry, full_info):
    script = [reply(tool("get_sales_metrics", {}, id=f"t{i}")) for i in range(5)]
    agent, llm = make_agent(registry, full_info, script, max_iterations=3)
    a = agent.ask("What is total revenue?")
    assert len(llm.calls) == 3
    assert "did not finish" in a.warnings[0]


def test_answer_without_tools_is_sent_back(registry, full_info):
    top = analytics.revenue_breakdown(registry.ctx.db, "category")[0]
    agent, llm = make_agent(registry, full_info, [
        reply(text("Probably Electronics.")),
        reply(tool("get_revenue_breakdown", {"dimension": "category"})),
        reply(text(f"{top['category']} generated the most revenue: {money(top['revenue'])}.")),
    ])
    a = agent.ask("Which category generated the most revenue?")
    assert llm.calls[1][-1]["content"] == CORRECTION_NO_TOOLS
    assert a.engine == "llm" and a.grounding.verified_numbers == 1


def test_chart_tool_returns_chart_to_user(registry, full_info):
    agent, llm = make_agent(registry, full_info, [
        reply(tool("generate_chart", {"chart": "monthly_revenue"})),
        reply(text("Here is monthly revenue as a chart.")),
    ])
    a = agent.ask("Show me monthly revenue as a chart.")
    assert len(a.charts) == 1 and a.charts[0]["chart"] == "monthly_revenue"
    assert '"rendered_to_user": true' in llm.calls[1][-1]["content"][0]["content"]


def test_history_is_sent_as_prior_turns(registry, full_info):
    agent, llm = make_agent(registry, full_info, [reply(tool("flag_unsupported", {"reason": "n/a"}))])
    agent.ask("And by region?", history=[{"question": "Revenue by category?", "answer": "Electronics leads."}])
    assert [m["role"] for m in llm.calls[0]] == ["user", "assistant", "user"]


def test_offline_mode_uses_rule_engine(registry, full_info):
    a = AnalystAgent(registry, full_info, llm=None).ask("Which category generated the most revenue?")
    assert a.engine == "rule_based" and "Offline mode" in a.warnings[0]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_invalid_questions_are_rejected(registry, full_info, question):
    with pytest.raises(InvalidInputError):
        AnalystAgent(registry, full_info, llm=None).ask(question)


def test_insight_writer_keeps_only_verified_rewrites(full_db, full_info):
    import json

    from app.agent.insight_writer import write_insights

    base = analytics.compute_insights(full_db, full_info)
    good, bad = base[0], base[1]
    payload = {"insights": [
        {"id": good["id"], "title": "Sharp October drop", "text": good["text"] + " Restock Electronics first."},
        {"id": bad["id"], "title": "Strong growth", "text": "Revenue is up 99.9% year over year."},
    ]}
    items, warnings = write_insights(FakeLLM([reply(text(json.dumps(payload)))]), base)
    assert (items[0]["source"], items[0]["title"]) == ("llm", "Sharp October drop")
    assert (items[1]["source"], items[1]["text"]) == ("template", bad["text"])
    assert "1 AI-written insight" in warnings[0]


def test_insight_writer_survives_llm_failure(full_db, full_info):
    from app.agent.insight_writer import write_insights

    base = analytics.compute_insights(full_db, full_info)
    items, warnings = write_insights(FakeLLM([LLMUnavailableError("timeout", "the API request timed out")]), base)
    assert all(i["source"] == "template" for i in items) and "timed out" in warnings[0]
