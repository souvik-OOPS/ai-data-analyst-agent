"""Chart specifications built from live query results.

The backend decides *what* to plot (data, chart type, labels, number format);
the frontend only decides *how it looks*. A spec is plain JSON:

    {"chart": "monthly_revenue", "type": "line", "title": ..., "series": [{"name", "x", "y"}], ...}
"""

from __future__ import annotations

import uuid
from typing import Literal

from app.analytics.customers import customer_trend
from app.analytics.periods import ALL_TIME, DateRange
from app.analytics.products import product_performance
from app.analytics.sales import monthly_sales, revenue_breakdown
from app.db import Database
from app.errors import DataNotFoundError, InvalidInputError

ChartName = Literal[
    "monthly_revenue",
    "monthly_orders",
    "aov_trend",
    "revenue_by_category",
    "revenue_by_region",
    "revenue_by_channel",
    "top_products",
    "bottom_products",
    "customer_trend",
]
ChartType = Literal["auto", "line", "bar", "pie"]

CHART_CATALOG: dict[str, dict] = {
    "monthly_revenue": {"title": "Monthly revenue", "types": ("line", "bar"), "default": "line"},
    "monthly_orders": {"title": "Monthly orders", "types": ("line", "bar"), "default": "bar"},
    "aov_trend": {"title": "Average order value by month", "types": ("line", "bar"), "default": "line"},
    "revenue_by_category": {"title": "Revenue by category", "types": ("bar", "pie"), "default": "bar"},
    "revenue_by_region": {"title": "Revenue by region", "types": ("bar", "pie"), "default": "bar"},
    "revenue_by_channel": {"title": "Revenue by channel", "types": ("bar", "pie"), "default": "bar"},
    "top_products": {"title": "Top products by revenue", "types": ("bar",), "default": "bar"},
    "bottom_products": {"title": "Lowest-revenue products", "types": ("bar",), "default": "bar"},
    "customer_trend": {"title": "New vs returning customers", "types": ("line", "bar"), "default": "line"},
}


def _period_label(period: DateRange) -> str:
    if not period.start and not period.end:
        return "All data"
    start = period.start.isoformat() if period.start else "start"
    end = period.end.isoformat() if period.end else "latest"
    return f"{start} to {end}"


def build_chart(
    db: Database,
    chart: str,
    chart_type: str = "auto",
    period: DateRange = ALL_TIME,
    limit: int = 10,
) -> dict:
    if chart not in CHART_CATALOG:
        raise InvalidInputError(f"Unknown chart '{chart}'. Available: {', '.join(CHART_CATALOG)}.")
    meta = CHART_CATALOG[chart]
    resolved = meta["default"] if chart_type in (None, "auto") else chart_type
    if resolved not in meta["types"]:
        raise InvalidInputError(f"Chart '{chart}' supports types: {', '.join(meta['types'])} (got '{chart_type}').")
    if not 3 <= int(limit) <= 30:
        raise InvalidInputError("limit must be between 3 and 30.")

    spec: dict = {
        "id": uuid.uuid4().hex[:12],
        "chart": chart,
        "type": resolved,
        "title": meta["title"],
        "subtitle": f"{_period_label(period)} · completed orders",
        "orientation": "v",
        "value_format": "currency",
        "x_label": "Month",
        "y_label": "Revenue (USD)",
        "series": [],
    }

    if chart in ("monthly_revenue", "monthly_orders", "aov_trend"):
        rows = monthly_sales(db, period)
        field, label, fmt = {
            "monthly_revenue": ("revenue", "Revenue (USD)", "currency"),
            "monthly_orders": ("orders", "Orders", "count"),
            "aov_trend": ("average_order_value", "Average order value (USD)", "currency"),
        }[chart]
        spec.update(y_label=label, value_format=fmt)
        spec["series"] = [{"name": label, "x": [r["month"] for r in rows], "y": [r[field] for r in rows]}]

    elif chart.startswith("revenue_by_"):
        dimension = chart.removeprefix("revenue_by_")
        rows = revenue_breakdown(db, dimension, period)  # type: ignore[arg-type]
        spec.update(orientation="h", x_label="Revenue (USD)", y_label=dimension.title())
        spec["series"] = [{"name": "Revenue", "x": [r[dimension] for r in rows], "y": [r["revenue"] for r in rows]}]

    elif chart in ("top_products", "bottom_products"):
        direction = "top" if chart == "top_products" else "bottom"
        result = product_performance(db, "revenue", direction, limit, None, period)
        spec.update(orientation="h", x_label="Revenue (USD)", y_label="Product",
                    title=f"{'Top' if direction == 'top' else 'Lowest'} {limit} products by revenue")
        spec["series"] = [{
            "name": "Revenue",
            "x": [p["product_name"] for p in result["products"]],
            "y": [p["revenue"] for p in result["products"]],
        }]

    elif chart == "customer_trend":
        rows = customer_trend(db, period)
        spec.update(y_label="Customers", value_format="count")
        months = [r["month"] for r in rows]
        spec["series"] = [
            {"name": "New customers", "x": months, "y": [r["new_customers"] for r in rows]},
            {"name": "Returning customers", "x": months, "y": [r["returning_customers"] for r in rows]},
        ]

    if not spec["series"] or not spec["series"][0]["x"]:
        raise DataNotFoundError(f"No data available for '{meta['title']}' in the selected period.")
    return spec


def chart_summary(spec: dict, max_points: int = 40) -> dict:
    """Compact version of a spec for the LLM: enough to describe the chart, no styling."""
    return {
        "chart_id": spec["id"],
        "rendered_to_user": True,
        "title": spec["title"],
        "type": spec["type"],
        "value_format": spec["value_format"],
        "series": [
            {"name": s["name"], "points": [{"x": x, "y": y} for x, y in list(zip(s["x"], s["y"], strict=True))[:max_points]]}
            for s in spec["series"]
        ],
    }
