"""Natural-language questions whose answers are verified independently with pandas.

These run the full question -> intent -> tool -> answer path of the offline engine
(no API key needed). ``test_live_llm.py`` asks the same questions to Claude.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.agent import AnalystAgent
from app.formatting import money, month_label, num, pct


@pytest.fixture(scope="module")
def agent(registry, full_info):
    return AnalystAgent(registry, full_info, llm=None)


def answered(agent, question):
    a = agent.ask(question)
    assert a.status == "answered", a.answer
    assert a.grounding.grounded, a.grounding.unverified_numbers
    return a


def test_top_5_products_by_revenue(agent, ref):
    expected = ref.lines.groupby("product_name")["revenue"].sum().sort_values(ascending=False).head(5)
    a = answered(agent, "What were the top 5 products by revenue?")
    positions = [a.answer.index(name) for name in expected.index]
    assert positions == sorted(positions), "products must be listed in revenue order"
    assert money(expected.iloc[0]) in a.answer
    assert a.answer.count("| Electronics |") + a.answer.count("| Home & Kitchen |") <= 5


def test_monthly_revenue(agent, ref):
    expected = ref.order_rev.groupby("month")["revenue"].sum()
    a = answered(agent, "Show monthly revenue.")
    for month, value in expected.items():
        assert f"| {month_label(month)} | {money(value)} |" in a.answer


def test_customers_not_purchased_recently(agent, ref, full_info):
    last = ref.order_rev.groupby("customer_id")["order_date"].max()
    cutoff = date.fromordinal(full_info.as_of.toordinal() - 90).isoformat()
    lapsed = last[last < cutoff]
    lifetime = ref.order_rev.groupby("customer_id")["revenue"].sum()
    most_valuable = lifetime[lapsed.index].sort_values(ascending=False).index[0]
    a = answered(agent, "Which customers have not purchased recently?")
    assert f"{num(len(lapsed))} customers have not purchased in the last 90 days" in a.answer
    assert most_valuable in a.answer


def test_category_with_most_revenue(agent, ref):
    by_cat = ref.lines.groupby("category")["revenue"].sum().sort_values(ascending=False)
    a = answered(agent, "Which category generated the most revenue?")
    assert a.answer.startswith(f"**{by_cat.index[0]} generated the most revenue")
    assert money(by_cat.iloc[0]) in a.answer


def test_why_did_revenue_decline_last_month(agent, ref):
    monthly = ref.order_rev.groupby("month")["revenue"].sum()
    oct_, sep = monthly["2026-10"], monthly["2026-09"]
    cat = ref.lines[ref.lines["month"].isin(["2026-09", "2026-10"])].pivot_table(
        index="category", columns="month", values="revenue", aggfunc="sum")
    biggest_drop = (cat["2026-10"] - cat["2026-09"]).idxmin()
    a = answered(agent, "Why did revenue decline last month?")
    assert pct(abs((oct_ - sep) / sep * 100)) in a.answer
    assert money(sep) in a.answer and money(oct_) in a.answer
    assert biggest_drop == "Electronics" and "Electronics revenue fell" in a.answer
    assert [c.tool for c in a.tool_calls] == ["analyze_revenue_change", "generate_chart"]


def test_monthly_revenue_chart(agent, ref):
    expected = ref.order_rev.groupby("month")["revenue"].sum()
    a = answered(agent, "Show me monthly revenue as a chart.")
    series = a.charts[0]["series"][0]
    assert series["x"] == list(expected.index)
    assert series["y"] == pytest.approx(list(expected.values), abs=0.01)


def test_orders_in_a_named_month(agent, ref):
    expected = ref.order_rev[ref.order_rev["month"] == "2026-09"]["order_id"].nunique()
    a = answered(agent, "How many orders did we have in September 2026?")
    assert f"from {num(expected)} completed orders" in a.answer


def test_lowest_performing_products_include_unsold(agent, ref):
    sold = set(ref.lines["product_name"])
    unsold = sorted(set(ref.products["product_name"]) - sold)
    a = answered(agent, "Which products are underperforming?")
    for name in unsold:
        assert name in a.answer


@pytest.mark.parametrize("question", [
    "What is our profit margin?",
    "How much did we spend on marketing last month?",
    "Forecast revenue for next month",
    "What's the weather in Paris?",
    "blorp zxcv",
])
def test_unsupported_questions(agent, question):
    a = agent.ask(question)
    assert a.status == "unsupported"
    assert "can't answer" in a.answer and a.tool_calls == []
