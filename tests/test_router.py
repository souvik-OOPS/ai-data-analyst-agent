"""Intent and parameter detection of the rule-based engine."""

from __future__ import annotations

import pytest

from app.agent.router import RuleBasedAnalyst


@pytest.fixture(scope="module")
def rules(registry, full_info):
    return RuleBasedAnalyst(registry, full_info)


@pytest.mark.parametrize("question,intent", [
    ("What were the top 5 products by revenue?", "products"),
    ("Show monthly revenue.", "monthly"),
    ("Which customers have not purchased recently?", "inactive_customers"),
    ("Which category generated the most revenue?", "breakdown"),
    ("Why did revenue decline last month?", "revenue_change"),
    ("Show me monthly revenue as a chart.", "chart"),
    ("Who are our top 10 customers?", "top_customers"),
    ("How many new customers did we get?", "customer_metrics"),
    ("What was the average order value?", "sales_metrics"),
    ("What date range does the data cover?", "dataset_info"),
    ("What is our profit margin?", "unsupported"),
    ("!!!", "invalid"),
    ("tell me something", "unknown"),
])
def test_intent(rules, question, intent):
    assert rules.detect(question).name == intent


def test_parameters(rules):
    p = rules.detect("top 3 best selling products last month").params
    assert p == {"direction": "top", "metric": "units", "limit": 3, "start_date": "2026-10-01",
                 "end_date": "2026-10-31"}
    assert rules.detect("worst products").params["direction"] == "bottom"
    assert rules.detect("customers inactive for 60 days").params["days"] == 60
    assert rules.detect("why did sales drop in September 2026?").params == {"month": "2026-09"}


@pytest.mark.parametrize("question,chart,chart_type", [
    ("plot revenue by category as a pie chart", "revenue_by_category", "pie"),
    ("graph new vs returning customers", "customer_trend", "auto"),
    ("chart the top products", "top_products", "auto"),
    ("bar chart of monthly orders", "monthly_orders", "bar"),
    ("pie chart of monthly revenue", "monthly_revenue", "auto"),  # pie not valid for a time series
])
def test_chart_selection(rules, question, chart, chart_type):
    params = rules.detect(question).params
    assert (params["chart"], params["chart_type"]) == (chart, chart_type)


@pytest.mark.parametrize("question,start,end", [
    ("revenue last 3 months", "2026-08-01", "2026-10-31"),
    ("revenue in 2025", "2025-01-01", "2025-12-31"),
    ("revenue in march 2026", "2026-03-01", "2026-03-31"),
    ("revenue this year", "2026-01-01", "2026-10-31"),
])
def test_period_parsing(rules, question, start, end):
    params = rules.detect(question).params
    assert (params["start_date"], params["end_date"]) == (start, end)
