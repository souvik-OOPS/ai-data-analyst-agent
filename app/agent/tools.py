"""The controlled tool layer the agent is allowed to use.

Every tool has a Pydantic input model (validated before execution, so a bad
argument becomes an error the model can correct rather than a crash), a
handler that calls the analytics layer, and a description written for the LLM.
The same registry backs both the Claude agent and the offline rule-based engine.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app import analytics
from app.analytics.periods import DatasetInfo, DateRange
from app.charts import ChartName, ChartType, build_chart, chart_summary
from app.db import SCHEMA_DESCRIPTION, Database
from app.errors import AppError
from app.sql_guard import run_read_only_query

logger = logging.getLogger(__name__)

MAX_LLM_CONTENT_CHARS = 16_000
UNSUPPORTED_TOOL = "flag_unsupported"


@dataclass
class ToolContext:
    db: Database
    info: DatasetInfo


# --------------------------------------------------------------------------- input models


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Period(_Strict):
    start_date: date | None = Field(None, description="Inclusive start date, YYYY-MM-DD. Omit for no lower bound.")
    end_date: date | None = Field(None, description="Inclusive end date, YYYY-MM-DD. Omit for no upper bound.")

    @model_validator(mode="after")
    def _check_order(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self

    def period(self) -> DateRange:
        return DateRange(self.start_date, self.end_date)


class NoInput(_Strict):
    pass


class BreakdownInput(_Period):
    dimension: Literal["category", "region", "channel"] = Field(..., description="Dimension to group revenue by.")


class ProductsInput(_Period):
    metric: Literal["revenue", "units"] = Field("revenue", description="Rank by revenue or by units sold.")
    direction: Literal["top", "bottom"] = Field("top", description="'top' = best performers, 'bottom' = worst.")
    limit: int = Field(10, ge=1, le=50, description="Number of products to return.")
    category: str | None = Field(None, description="Optional category filter, e.g. 'Electronics'.")


class CustomerMetricsInput(_Period):
    inactive_days: int = Field(90, ge=7, le=730, description="Days without a purchase that counts as inactive.")


class TopCustomersInput(_Period):
    limit: int = Field(10, ge=1, le=50)


class InactiveCustomersInput(_Strict):
    days: int = Field(90, ge=7, le=730, description="No purchase within this many days of the dataset's last date.")
    limit: int = Field(20, ge=1, le=100)


class RevenueChangeInput(_Strict):
    month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$",
                              description="Month to explain, YYYY-MM. Omit for the latest complete month.")
    compare_to: str | None = Field(None, pattern=r"^\d{4}-\d{2}$",
                                   description="Baseline month, YYYY-MM. Omit for the month before `month`.")


class ChartInput(_Period):
    chart: ChartName = Field(..., description="Which chart to build.")
    chart_type: ChartType = Field("auto", description="'auto' picks the best type; pie only for breakdowns.")
    limit: int = Field(10, ge=3, le=30, description="Number of products for product charts.")


class SqlInput(_Strict):
    query: str = Field(..., min_length=1, max_length=2000, description="A single read-only SELECT statement.")


class UnsupportedInput(_Strict):
    reason: str = Field(..., min_length=1, max_length=500, description="Why the dataset cannot answer this.")
    suggestion: str | None = Field(None, max_length=300, description="A related question that can be answered.")


# --------------------------------------------------------------------------- handlers


def _dataset_info(ctx: ToolContext, _: NoInput) -> dict:
    return {
        **ctx.info.to_dict(),
        "schema": SCHEMA_DESCRIPTION,
        "definitions": {
            "revenue": "sum of quantity * unit_price * (1 - discount) over completed orders (net of discounts)",
            "orders": "count of completed orders; cancelled and returned orders are excluded from revenue",
            "average_order_value": "revenue / orders",
            "new_customer": "first-ever completed order falls in the period",
            "inactive_customer": "has purchased before, but not within N days of the as_of_date",
            "relative_dates": f"'last month' = {ctx.info.latest_full_month}; all relative dates are anchored to "
            f"as_of_date {ctx.info.as_of.isoformat()}, not today's date",
        },
    }


def _sales(ctx: ToolContext, a: _Period) -> dict:
    return analytics.sales_metrics(ctx.db, a.period())


def _monthly(ctx: ToolContext, a: _Period) -> dict:
    return {"months": analytics.monthly_sales(ctx.db, a.period())}


def _breakdown(ctx: ToolContext, a: BreakdownInput) -> dict:
    return {"dimension": a.dimension, "period": a.period().describe(),
            "rows": analytics.revenue_breakdown(ctx.db, a.dimension, a.period())}


def _products(ctx: ToolContext, a: ProductsInput) -> dict:
    return analytics.product_performance(ctx.db, a.metric, a.direction, a.limit, a.category, a.period())


def _customer_metrics(ctx: ToolContext, a: CustomerMetricsInput) -> dict:
    return analytics.customer_metrics(ctx.db, ctx.info.as_of, a.period(), a.inactive_days)


def _top_customers(ctx: ToolContext, a: TopCustomersInput) -> dict:
    return {"period": a.period().describe(), "customers": analytics.top_customers(ctx.db, a.limit, a.period())}


def _inactive(ctx: ToolContext, a: InactiveCustomersInput) -> dict:
    return analytics.inactive_customers(ctx.db, ctx.info.as_of, a.days, a.limit)


def _customer_trend(ctx: ToolContext, a: _Period) -> dict:
    return {"months": analytics.customer_trend(ctx.db, a.period())}


def _revenue_change(ctx: ToolContext, a: RevenueChangeInput) -> dict:
    return analytics.analyze_revenue_change(ctx.db, ctx.info, a.month, a.compare_to)


def _chart(ctx: ToolContext, a: ChartInput) -> dict:
    return build_chart(ctx.db, a.chart, a.chart_type, a.period(), a.limit)


def _sql(ctx: ToolContext, a: SqlInput) -> dict:
    return run_read_only_query(ctx.db, a.query)


def _unsupported(ctx: ToolContext, a: UnsupportedInput) -> dict:
    return {"acknowledged": True}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], dict]
    returns_chart: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": _clean_schema(self.input_model.model_json_schema())}


def _clean_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


TOOLS: list[ToolSpec] = [
    ToolSpec("get_dataset_info",
             "Describe the dataset: date coverage, as-of date, latest complete month, categories, regions, channels, "
             "table schemas and metric definitions. Use it to resolve relative dates or check what data exists.",
             NoInput, _dataset_info),
    ToolSpec("get_sales_metrics",
             "Headline KPIs for a period: revenue, orders, average order value, units, purchasing customers, "
             "discounts and cancellation rate. Omit dates for all-time figures.",
             _Period, _sales),
    ToolSpec("get_monthly_revenue",
             "Month-by-month revenue, orders, AOV, units, customers and month-over-month revenue change (%).",
             _Period, _monthly),
    ToolSpec("get_revenue_breakdown",
             "Revenue, orders, units and share of total grouped by category, region or channel, sorted by revenue.",
             BreakdownInput, _breakdown),
    ToolSpec("get_top_products",
             "Rank products by revenue or units sold. direction='top' for best sellers / highest revenue, "
             "'bottom' for low performers (includes products with zero sales). Optional category filter.",
             ProductsInput, _products),
    ToolSpec("get_customer_metrics",
             "Customer KPIs for a period: registered, active, new, returning and repeat customers, repeat rate, "
             "revenue per customer, plus the count of inactive customers.",
             CustomerMetricsInput, _customer_metrics),
    ToolSpec("get_top_customers", "Highest-revenue customers in a period with order counts and last order date.",
             TopCustomersInput, _top_customers),
    ToolSpec("get_inactive_customers",
             "Customers who have bought before but not recently (no purchase within `days` of the dataset's last "
             "date), most valuable first, with totals. Use for 'who hasn't purchased recently' or churn questions.",
             InactiveCustomersInput, _inactive),
    ToolSpec("get_customer_trend", "Per month: active, new and returning customers and the revenue from each group.",
             _Period, _customer_trend),
    ToolSpec("analyze_revenue_change",
             "Multi-step diagnosis of why revenue changed between two months: KPI comparison, order-volume vs "
             "order-value decomposition, category and region contributions, product movers (including products whose "
             "sales nearly stopped), new vs returning customers and cancellation rate. Returns evidence and "
             "plain-language findings. Use for 'why did revenue drop/grow' questions.",
             RevenueChangeInput, _revenue_change),
    ToolSpec("generate_chart",
             "Build a chart from live data and show it to the user. Use whenever the user asks for a chart, graph, "
             "plot or visual. Charts: monthly_revenue, monthly_orders, aov_trend, revenue_by_category, "
             "revenue_by_region, revenue_by_channel, top_products, bottom_products, customer_trend. "
             "Returns the plotted data points so you can describe them.",
             ChartInput, _chart, returns_chart=True),
    ToolSpec("run_sql",
             "Run one read-only SQLite SELECT for questions the other tools cannot answer. Allowed relations: "
             "customers, products, orders, order_items, and the views order_totals, sales, customer_summary "
             "(see get_dataset_info for columns). Revenue must come from completed orders only. Max 200 rows.",
             SqlInput, _sql),
    ToolSpec(UNSUPPORTED_TOOL,
             "Call this instead of answering when the question cannot be answered from this dataset (for example "
             "profit or margins, marketing spend, web traffic, inventory levels, forecasts, or topics unrelated to "
             "the business). Explain the reason and optionally suggest an answerable question.",
             UnsupportedInput, _unsupported),
]


# --------------------------------------------------------------------------- execution


@dataclass
class ToolResult:
    name: str
    input: dict
    ok: bool
    output: dict | None = None
    error: str | None = None
    duration_ms: float = 0.0
    chart: dict | None = None
    extra: dict = field(default_factory=dict)

    def content_for_llm(self) -> str:
        payload = self.output if self.ok else {"error": self.error}
        text = json.dumps(payload, default=str)
        if len(text) > MAX_LLM_CONTENT_CHARS:
            text = json.dumps({"truncated": True, "note": "Result too large; narrow the query.",
                               "preview": text[:MAX_LLM_CONTENT_CHARS]})
        return text

    def preview(self, limit: int = 600) -> str:
        text = self.content_for_llm()
        return text if len(text) <= limit else text[:limit] + " ..."


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "input"
        parts.append(f"{loc}: {err['msg']}")
    return "Invalid arguments - " + "; ".join(parts)


class ToolRegistry:
    def __init__(self, ctx: ToolContext, tools: list[ToolSpec] | None = None):
        self.ctx = ctx
        self._tools = {t.name: t for t in (tools or TOOLS)}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, raw_input: dict | None) -> ToolResult:
        started = time.perf_counter()
        raw_input = dict(raw_input or {})

        def done(**kwargs) -> ToolResult:
            result = ToolResult(name=name, input=raw_input, duration_ms=round((time.perf_counter() - started) * 1000, 1),
                                **kwargs)
            logger.info("tool=%s ok=%s %.1fms %s", name, result.ok, result.duration_ms, result.error or "")
            return result

        spec = self._tools.get(name)
        if spec is None:
            return done(ok=False, error=f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}.")
        try:
            args = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            return done(ok=False, error=_format_validation_error(exc))
        try:
            output = spec.handler(self.ctx, args)
        except AppError as exc:
            return done(ok=False, error=exc.message)
        except sqlite3.Error as exc:
            logger.exception("Database error in tool %s", name)
            return done(ok=False, error=f"Database error: {exc}")
        except Exception:  # never let a tool crash the agent loop
            logger.exception("Unexpected error in tool %s", name)
            return done(ok=False, error="Internal error while running this tool.")
        if spec.returns_chart:
            return done(ok=True, output=chart_summary(output), chart=output)
        return done(ok=True, output=output)
