"""HTTP API: success paths, input validation and error envelopes."""

from __future__ import annotations

import pytest


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["llm"]["mode"] == "rule_based"
    assert body["database"]["orders"] > 10000


def test_dashboard_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DDL Labs Analyst" in r.text


def test_dataset_includes_quality_report(client):
    body = client.get("/api/dataset").json()
    assert body["coverage"]["latest_full_month"] == "2026-10"
    assert body["ingestion_report"]["tables"]["orders"]["duplicates_removed"] == 40


def test_kpis_with_previous_period(client):
    body = client.get("/api/kpis", params={"start_date": "2026-10-01", "end_date": "2026-10-31"}).json()
    assert body["previous_period"] == {"start_date": "2026-09-01", "end_date": "2026-09-30"}
    assert body["change_pct"]["revenue"] < 0


def test_kpis_all_time_has_no_comparison(client):
    body = client.get("/api/kpis").json()
    assert body["current"]["orders"] > 0 and body["previous"] is None


@pytest.mark.parametrize("path,params", [
    ("/api/kpis", {"start_date": "2026-11-01", "end_date": "2026-10-01"}),
    ("/api/kpis", {"start_date": "not-a-date"}),
    ("/api/products", {"limit": 0}),
    ("/api/products", {"metric": "profit"}),
    ("/api/products", {"category": "Toys"}),
    ("/api/metrics/breakdown", {"dimension": "planet"}),
    ("/api/customers/inactive", {"days": 3}),
    ("/api/charts/monthly_revenue", {"chart_type": "pie"}),
    ("/api/charts/not_a_chart", {}),
    ("/api/analysis/revenue-change", {"month": "October"}),
])
def test_invalid_input_returns_422(client, path, params):
    r = client.get(path, params=params)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_input" and r.json()["error"]["message"]


def test_unknown_month_returns_404(client):
    r = client.get("/api/analysis/revenue-change", params={"month": "2019-01"})
    assert r.status_code == 404 and "dataset covers" in r.json()["error"]["message"]


def test_chart_endpoint(client):
    spec = client.get("/api/charts/top_products", params={"limit": 5}).json()
    assert spec["type"] == "bar" and len(spec["series"][0]["x"]) == 5


def test_metric_endpoints(client):
    assert client.get("/api/metrics/sales").json()["revenue"] > 0
    assert len(client.get("/api/metrics/monthly").json()["months"]) == 24
    assert client.get("/api/metrics/breakdown", params={"dimension": "region"}).json()["rows"]
    assert client.get("/api/metrics/customers").json()["repeat_customers"] > 0
    assert len(client.get("/api/customers/top", params={"limit": 3}).json()["customers"]) == 3
    assert client.get("/api/customers/inactive").json()["inactive_customer_count"] > 0
    assert client.get("/api/customers/trend").json()["months"]
    assert client.get("/api/analysis/revenue-change").json()["direction"] == "decline"


def test_insights(client):
    body = client.get("/api/insights").json()
    assert body["engine"] == "template" and len(body["insights"]) >= 6


def test_ask(client):
    body = client.post("/api/ask", json={"question": "Which category generated the most revenue?"}).json()
    assert body["status"] == "answered" and body["engine"] == "rule_based"
    assert body["grounding"]["grounded"] is True and body["tool_calls"][0]["tool"] == "get_revenue_breakdown"


def test_ask_unsupported(client):
    body = client.post("/api/ask", json={"question": "What is our profit margin?"}).json()
    assert body["status"] == "unsupported"


@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": 42},
    {"question": "ok", "unexpected": True},
    {"question": "ok", "history": [{"question": "q"}]},
])
def test_ask_validation(client, payload):
    r = client.post("/api/ask", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_query_endpoint(client):
    ok = client.post("/api/query", json={"query": "SELECT COUNT(*) AS n FROM orders"})
    assert ok.status_code == 200 and ok.json()["rows"][0]["n"] > 0
    bad = client.post("/api/query", json={"query": "DROP TABLE orders"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "query_rejected"
