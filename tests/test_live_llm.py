"""Optional end-to-end tests against the real LLM API configured in .env (Gemini or Claude).

Skipped unless an API key is configured. Run with:  pytest -m live
Note: each question uses 2-4 model requests; free-tier quotas are small.
Expected values come from the analytics layer, which test_analytics.py verifies against pandas.
"""

from __future__ import annotations

import pytest

from app import analytics
from app.agent.agent import AnalystAgent
from app.config import Settings
from app.services import create_llm

SETTINGS = Settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not SETTINGS.llm_configured, reason="no LLM API key configured in .env"),
]


@pytest.fixture(scope="module")
def agent(registry, full_info):
    return AnalystAgent(registry, full_info, create_llm(SETTINGS))


def llm_answer(agent, question):
    a = agent.ask(question)
    assert a.engine == "llm", a.warnings
    assert a.grounding is None or a.grounding.grounded, a.grounding.unverified_numbers
    return a


def test_top_products(agent, full_db):
    expected = [p["product_name"] for p in analytics.product_performance(full_db, limit=5)["products"]]
    a = llm_answer(agent, "What were the top 5 products by revenue?")
    assert all(name in a.answer for name in expected)


def test_top_category(agent, full_db):
    top = analytics.revenue_breakdown(full_db, "category")[0]["category"]
    assert top in llm_answer(agent, "Which category generated the most revenue?").answer


def test_revenue_decline_diagnosis(agent):
    a = llm_answer(agent, "Why did revenue decline last month?")
    assert "analyze_revenue_change" in [c.tool for c in a.tool_calls]
    assert "Electronics" in a.answer


def test_chart_request(agent):
    a = llm_answer(agent, "Show me monthly revenue as a chart.")
    assert a.charts and a.charts[0]["chart"] == "monthly_revenue"


def test_unsupported(agent):
    assert agent.ask("What is our profit margin?").status == "unsupported"
