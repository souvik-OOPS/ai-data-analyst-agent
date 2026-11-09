"""Business calculations.

Part 1 checks hand-computed answers on the tiny dataset.
Part 2 cross-checks the SQL analytics on the full sample against independent pandas code.
"""

from __future__ import annotations

from datetime import date

import pytest

from app import analytics
from app.analytics.periods import DateRange, previous_month, shift_month
from app.analytics.sales import previous_period
from app.charts import build_chart
from app.errors import DataNotFoundError, InvalidInputError

# --------------------------------------------------------------------------- tiny dataset (hand-computed)


def test_sales_kpis(tiny_db):
    m = analytics.sales_metrics(tiny_db)
    assert m["revenue"] == 1010.0
    assert m["orders"] == 4
    assert m["average_order_value"] == 252.5
    assert m["units_sold"] == 10
    assert m["customers"] == 3
    assert m["cancelled_orders"] == 2
    assert m["cancellation_rate_pct"] == 33.3


def test_monthly_revenue(tiny_db):
    months = {m["month"]: m for m in analytics.monthly_sales(tiny_db)}
    assert {k: v["revenue"] for k, v in months.items()} == {"2025-01": 30.0, "2025-02": 930.0, "2025-03": 50.0}
    assert months["2025-02"]["orders"] == 2
    assert months["2025-02"]["revenue_change_pct"] == 3000.0


def test_revenue_by_category(tiny_db):
    rows = analytics.revenue_breakdown(tiny_db, "category")
    assert [(r["category"], r["revenue"], r["share_pct"]) for r in rows] == [("Gadgets", 960.0, 95.0),
                                                                              ("Books", 50.0, 5.0)]


def test_product_rankings(tiny_db):
    top = analytics.product_performance(tiny_db, "revenue", "top", 3)["products"]
    assert [(p["product_name"], p["revenue"]) for p in top] == [("Gizmo", 900.0), ("Widget", 60.0), ("Book A", 50.0)]
    by_units = analytics.product_performance(tiny_db, "units", "top", 2)["products"]
    assert [(p["product_name"], p["units_sold"]) for p in by_units] == [("Widget", 6), ("Book A", 3)]
    bottom = analytics.product_performance(tiny_db, "revenue", "bottom", 1)["products"]
    assert bottom[0]["product_name"] == "Mystery Item" and bottom[0]["revenue"] == 0.0  # never sold


def test_product_category_filter_is_validated(tiny_db):
    gadgets = analytics.product_performance(tiny_db, category="gadgets")["products"]
    assert {p["category"] for p in gadgets} == {"Gadgets"}
    with pytest.raises(InvalidInputError, match="Valid categories"):
        analytics.product_performance(tiny_db, category="Toys")


def test_customer_metrics(tiny_db, tiny_info):
    c = analytics.customer_metrics(tiny_db, tiny_info.as_of)
    assert c["total_registered_customers"] == 4
    assert c["never_purchased_customers"] == 1
    assert c["repeat_customers"] == 1
    assert c["repeat_customer_rate_pct"] == 33.3
    assert c["repeat_customer_revenue_share_pct"] == 92.1  # C1: 930 / 1010
    feb = analytics.customer_metrics(tiny_db, tiny_info.as_of, DateRange(date(2025, 2, 1), date(2025, 2, 28)))
    assert (feb["active_customers"], feb["new_customers"], feb["returning_customers"]) == (2, 1, 1)


def test_inactive_customers(tiny_db, tiny_info):
    assert tiny_info.as_of == date(2025, 3, 20)
    assert analytics.inactive_customers(tiny_db, tiny_info.as_of, days=90)["inactive_customer_count"] == 0
    r = analytics.inactive_customers(tiny_db, tiny_info.as_of, days=30)
    assert r["inactive_customer_count"] == 1
    assert r["customers"][0]["customer_id"] == "C1"
    assert r["customers"][0]["lifetime_revenue"] == 930.0
    assert r["customers"][0]["days_since_last_order"] == 33
    with pytest.raises(InvalidInputError):
        analytics.inactive_customers(tiny_db, tiny_info.as_of, days=1)


def test_latest_full_month_ignores_partial_month(tiny_info):
    # Data ends 2025-03-20, so March is incomplete and "last month" is February.
    assert tiny_info.latest_full_month == "2025-02"


def test_revenue_change_reports_increase_instead_of_inventing_a_decline(tiny_db, tiny_info):
    a = analytics.analyze_revenue_change(tiny_db, tiny_info)
    assert (a["month"], a["compare_to"], a["direction"]) == ("2025-02", "2025-01", "increase")
    assert a["revenue_change"] == 900.0
    assert a["findings"][0].startswith("Revenue did not decline")
    d = a["decomposition"]
    assert d["order_volume_effect"] + d["order_value_effect"] == pytest.approx(a["revenue_change"], abs=0.02)


def test_revenue_change_input_validation(tiny_db, tiny_info):
    with pytest.raises(DataNotFoundError):
        analytics.analyze_revenue_change(tiny_db, tiny_info, "2020-01")
    with pytest.raises(InvalidInputError):
        analytics.analyze_revenue_change(tiny_db, tiny_info, "2025-13")
    with pytest.raises(InvalidInputError):
        analytics.analyze_revenue_change(tiny_db, tiny_info, "2025-02", "2025-02")


def test_period_helpers(tiny_info):
    assert shift_month("2025-01", -1) == "2024-12" and previous_month("2025-03") == "2025-02"
    with pytest.raises(InvalidInputError):
        DateRange(date(2025, 2, 1), date(2025, 1, 1))
    assert previous_period(DateRange(), tiny_info) is None
    prev = previous_period(DateRange(date(2025, 3, 1), date(2025, 3, 31)), tiny_info)
    assert (prev.start, prev.end) == (date(2025, 2, 1), date(2025, 2, 28))
    prev = previous_period(DateRange(date(2025, 2, 11), date(2025, 2, 20)), tiny_info)
    assert (prev.start, prev.end) == (date(2025, 2, 1), date(2025, 2, 10))
    # Data starts 2025-01-10, so January is partial: comparing February with it would mislead.
    assert previous_period(DateRange(date(2025, 2, 1), date(2025, 2, 28)), tiny_info) is None


def test_chart_spec_uses_query_results(tiny_db):
    spec = build_chart(tiny_db, "monthly_revenue")
    assert spec["series"][0]["x"] == ["2025-01", "2025-02", "2025-03"]
    assert spec["series"][0]["y"] == [30.0, 930.0, 50.0]
    with pytest.raises(InvalidInputError):
        build_chart(tiny_db, "monthly_revenue", "pie")
    with pytest.raises(DataNotFoundError):
        build_chart(tiny_db, "monthly_revenue", period=DateRange(date(2030, 1, 1), date(2030, 1, 31)))


# --------------------------------------------------------------------------- full sample vs independent pandas


def test_total_revenue_matches_pandas(full_db, ref):
    m = analytics.sales_metrics(full_db)
    assert m["revenue"] == pytest.approx(ref.lines["revenue"].sum(), abs=0.01)
    assert m["orders"] == ref.order_rev["order_id"].nunique()
    assert m["average_order_value"] == pytest.approx(ref.lines["revenue"].sum() / ref.order_rev["order_id"].nunique(),
                                                     abs=0.01)


def test_monthly_revenue_matches_pandas(full_db, ref):
    expected = ref.order_rev.groupby("month")["revenue"].sum()
    actual = {m["month"]: m["revenue"] for m in analytics.monthly_sales(full_db)}
    assert list(actual) == list(expected.index)
    for month, value in expected.items():
        assert actual[month] == pytest.approx(value, abs=0.01)


def test_category_revenue_matches_pandas(full_db, ref):
    expected = ref.lines.groupby("category")["revenue"].sum().sort_values(ascending=False)
    rows = analytics.revenue_breakdown(full_db, "category")
    assert [r["category"] for r in rows] == list(expected.index)
    assert rows[0]["revenue"] == pytest.approx(expected.iloc[0], abs=0.01)


def test_top_products_match_pandas(full_db, ref):
    expected = ref.lines.groupby("product_name")["revenue"].sum().sort_values(ascending=False).head(5)
    top = analytics.product_performance(full_db, "revenue", "top", 5)["products"]
    assert [p["product_name"] for p in top] == list(expected.index)


def test_new_customers_per_month_match_pandas(full_db, ref):
    firsts = ref.order_rev.groupby("customer_id")["month"].min().value_counts()
    trend = {r["month"]: r["new_customers"] for r in analytics.customer_trend(full_db)}
    assert trend == firsts.to_dict()


def test_inactive_customers_match_pandas(full_db, full_info, ref):
    last = ref.order_rev.groupby("customer_id")["order_date"].max()
    cutoff = date.fromordinal(full_info.as_of.toordinal() - 90).isoformat()
    expected = int((last < cutoff).sum())
    assert analytics.inactive_customers(full_db, full_info.as_of, 90)["inactive_customer_count"] == expected


def test_planted_october_story_is_detected(full_db, full_info):
    a = analytics.analyze_revenue_change(full_db, full_info)
    assert (a["month"], a["compare_to"], a["direction"]) == ("2026-10", "2026-09", "decline")
    assert a["categories"][0]["category"] == "Electronics"
    stalled = {p["product_name"] for p in a["products_with_sales_nearly_stopped"]}
    assert {"Nimbus 27in 4K Monitor", "Spark 10in Tablet"} <= stalled
    assert a["customers"]["new_customers"]["change"] < 0
    assert a["headline"]["cancellation_rate_pct"]["current"] > a["headline"]["cancellation_rate_pct"]["previous"]


def test_insights_are_backed_by_evidence(full_db, full_info):
    insights = analytics.compute_insights(full_db, full_info)
    assert len(insights) >= 6
    assert all(i["evidence"] and i["text"] for i in insights)
