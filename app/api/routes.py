"""HTTP API. Routes stay thin: validate input, call the analytics/agent layer, return JSON."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from fastapi import APIRouter, Path, Query, Request

from app import __version__, analytics
from app.analytics.periods import DateRange
from app.analytics.sales import kpis_with_comparison
from app.api.schemas import AskRequest, SqlRequest
from app.charts import CHART_CATALOG, build_chart
from app.db import SCHEMA_DESCRIPTION
from app.services import Services
from app.sql_guard import run_read_only_query

router = APIRouter(prefix="/api")

StartDate = Query(None, description="Inclusive start date (YYYY-MM-DD)")
EndDate = Query(None, description="Inclusive end date (YYYY-MM-DD)")


def _services(request: Request) -> Services:
    return request.app.state.services


def _period(start_date: date | None, end_date: date | None) -> DateRange:
    return DateRange(start_date, end_date)  # raises InvalidInputError (422) if start > end


@router.get("/health", summary="Service health, database status and LLM mode")
def health(request: Request) -> dict:
    s = _services(request)
    return {
        "status": "ok",
        "version": __version__,
        "database": {"path": s.db.path.name, "orders": s.info.orders, "customers": s.info.customers,
                     "products": s.info.products},
        "llm": {"mode": s.agent.mode, "configured": s.llm is not None,
                "provider": s.llm.provider if s.llm else None, "model": s.llm.model if s.llm else None},
    }


@router.get("/dataset", summary="Dataset coverage, schema, definitions and data-quality report")
def dataset(request: Request) -> dict:
    s = _services(request)
    raw = s.db.scalar("SELECT value FROM metadata WHERE key = 'ingestion_report'")
    return {
        "coverage": s.info.to_dict(),
        "schema": SCHEMA_DESCRIPTION,
        "ingestion_report": json.loads(raw) if raw else None,
        "charts": {name: meta["title"] for name, meta in CHART_CATALOG.items()},
    }


@router.get("/kpis", summary="Revenue, orders, customers and AOV with comparison to the previous period")
def kpis(request: Request, start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    s = _services(request)
    return kpis_with_comparison(s.db, s.info, _period(start_date, end_date))


@router.get("/metrics/sales", summary="Headline sales metrics")
def sales(request: Request, start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return analytics.sales_metrics(_services(request).db, _period(start_date, end_date))


@router.get("/metrics/monthly", summary="Monthly revenue, orders and AOV")
def monthly(request: Request, start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return {"months": analytics.monthly_sales(_services(request).db, _period(start_date, end_date))}


@router.get("/metrics/breakdown", summary="Revenue by category, region or channel")
def breakdown(request: Request, dimension: Literal["category", "region", "channel"] = "category",
              start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    rows = analytics.revenue_breakdown(_services(request).db, dimension, _period(start_date, end_date))
    return {"dimension": dimension, "rows": rows}


@router.get("/metrics/customers", summary="Customer KPIs: total, new, returning, repeat, inactive")
def customers(request: Request, inactive_days: int = Query(90, ge=7, le=730),
              start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    s = _services(request)
    return analytics.customer_metrics(s.db, s.info.as_of, _period(start_date, end_date), inactive_days)


@router.get("/products", summary="Top or bottom products by revenue or units")
def products(request: Request, metric: Literal["revenue", "units"] = "revenue",
             direction: Literal["top", "bottom"] = "top", limit: int = Query(10, ge=1, le=100),
             category: str | None = Query(None, max_length=60),
             start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return analytics.product_performance(_services(request).db, metric, direction, limit, category,
                                         _period(start_date, end_date))


@router.get("/customers/top", summary="Highest-revenue customers")
def customers_top(request: Request, limit: int = Query(10, ge=1, le=100),
                  start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return {"customers": analytics.top_customers(_services(request).db, limit, _period(start_date, end_date))}


@router.get("/customers/inactive", summary="Customers who have not purchased recently")
def customers_inactive(request: Request, days: int = Query(90, ge=7, le=730),
                       limit: int = Query(20, ge=1, le=200)) -> dict:
    s = _services(request)
    return analytics.inactive_customers(s.db, s.info.as_of, days, limit)


@router.get("/customers/trend", summary="New vs returning customers per month")
def customers_trend(request: Request, start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return {"months": analytics.customer_trend(_services(request).db, _period(start_date, end_date))}


@router.get("/charts/{chart}", summary="Chart specification built from live data")
def chart(request: Request, chart: str = Path(..., max_length=40),
          chart_type: Literal["auto", "line", "bar", "pie"] = "auto", limit: int = Query(10, ge=3, le=30),
          start_date: date | None = StartDate, end_date: date | None = EndDate) -> dict:
    return build_chart(_services(request).db, chart, chart_type, _period(start_date, end_date), limit)


@router.get("/analysis/revenue-change", summary="Multi-step diagnosis of a month-over-month revenue change")
def revenue_change(request: Request,
                   month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM; default latest"),
                   compare_to: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")) -> dict:
    s = _services(request)
    return analytics.analyze_revenue_change(s.db, s.info, month, compare_to)


@router.get("/insights", summary="Data-backed business insights (optionally rewritten by the LLM)")
def insights(request: Request, ai: bool = True) -> dict:
    return _services(request).insights(use_llm=ai)


@router.post("/ask", summary="Ask the AI analyst a natural-language question")
def ask(request: Request, body: AskRequest) -> dict:
    s = _services(request)
    answer = s.agent.ask(body.question, [t.model_dump() for t in body.history])
    return answer.model_dump()


@router.post("/query", summary="Run a guarded read-only SQL query")
def query(request: Request, body: SqlRequest) -> dict:
    return run_read_only_query(_services(request).db, body.query)
