"""Sales metrics: revenue, orders, AOV, monthly trend and revenue breakdowns."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from app.analytics.periods import ALL_TIME, DatasetInfo, DateRange, month_bounds, shift_month
from app.db import Database
from app.formatting import pct_change, r2, share

Dimension = Literal["category", "region", "channel"]

# Whitelisted SQL fragments per dimension: never interpolate user input into SQL.
_BREAKDOWN_SQL: dict[str, str] = {
    "category": """
        SELECT category AS label, SUM(line_revenue) AS revenue,
               COUNT(DISTINCT order_id) AS orders, SUM(quantity) AS units
        FROM sales WHERE order_date BETWEEN ? AND ?
        GROUP BY category ORDER BY revenue DESC""",
    "region": """
        SELECT region AS label, SUM(order_revenue) AS revenue, COUNT(*) AS orders, SUM(units) AS units
        FROM order_totals WHERE status = 'completed' AND order_date BETWEEN ? AND ?
        GROUP BY region ORDER BY revenue DESC""",
    "channel": """
        SELECT channel AS label, SUM(order_revenue) AS revenue, COUNT(*) AS orders, SUM(units) AS units
        FROM order_totals WHERE status = 'completed' AND order_date BETWEEN ? AND ?
        GROUP BY channel ORDER BY revenue DESC""",
}


def sales_metrics(db: Database, period: DateRange = ALL_TIME) -> dict:
    """Headline KPIs for completed orders in the period."""
    row = db.query_one(
        """
        SELECT COUNT(*)                          AS orders,
               COALESCE(SUM(order_revenue), 0)   AS revenue,
               COALESCE(SUM(units), 0)           AS units,
               COUNT(DISTINCT customer_id)       AS customers,
               COALESCE(SUM(discount_amount), 0) AS discounts,
               MIN(order_date)                   AS first_date,
               MAX(order_date)                   AS last_date
        FROM order_totals
        WHERE status = 'completed' AND order_date BETWEEN ? AND ?
        """,
        period.params(),
    )
    status = db.query_one(
        """
        SELECT COUNT(*) AS all_orders,
               COALESCE(SUM(status = 'cancelled'), 0) AS cancelled,
               COALESCE(SUM(status = 'returned'), 0)  AS returned
        FROM order_totals WHERE order_date BETWEEN ? AND ?
        """,
        period.params(),
    )
    orders = row["orders"]
    result = {
        "period": {**period.describe(), "data_from": row["first_date"], "data_to": row["last_date"]},
        "revenue": r2(row["revenue"]),
        "orders": orders,
        "average_order_value": r2(row["revenue"] / orders) if orders else None,
        "units_sold": row["units"],
        "customers": row["customers"],
        "discounts_given": r2(row["discounts"]),
        "cancelled_orders": status["cancelled"],
        "returned_orders": status["returned"],
        "cancellation_rate_pct": share(status["cancelled"], status["all_orders"]),
    }
    if not orders:
        result["note"] = "No completed orders in this period."
    return result


def monthly_sales(db: Database, period: DateRange = ALL_TIME) -> list[dict]:
    rows = db.query(
        """
        SELECT month,
               COUNT(*)                    AS orders,
               SUM(order_revenue)          AS revenue,
               SUM(units)                  AS units,
               COUNT(DISTINCT customer_id) AS customers
        FROM order_totals
        WHERE status = 'completed' AND order_date BETWEEN ? AND ?
        GROUP BY month ORDER BY month
        """,
        period.params(),
    )
    out, prev = [], None
    for row in rows:
        out.append({
            "month": row["month"],
            "revenue": r2(row["revenue"]),
            "orders": row["orders"],
            "average_order_value": r2(row["revenue"] / row["orders"]),
            "units_sold": row["units"],
            "customers": row["customers"],
            "revenue_change_pct": pct_change(row["revenue"], prev["revenue"]) if prev else None,
        })
        prev = row
    return out


def revenue_breakdown(db: Database, dimension: Dimension, period: DateRange = ALL_TIME) -> list[dict]:
    if dimension not in _BREAKDOWN_SQL:
        raise ValueError(f"Unsupported dimension: {dimension}")
    rows = db.query(_BREAKDOWN_SQL[dimension], period.params())
    total = sum(r["revenue"] for r in rows)
    return [
        {
            dimension: r["label"],
            "revenue": r2(r["revenue"]),
            "orders": r["orders"],
            "units_sold": r["units"],
            "share_pct": share(r["revenue"], total),
        }
        for r in rows
    ]


def previous_period(period: DateRange, info: DatasetInfo) -> DateRange | None:
    """The equally long period immediately before ``period`` (whole months when the range is whole months).

    Returns None for open-ended ranges or when the previous period starts before the data does,
    because a comparison against a partial period would be misleading.
    """
    if not period.start or not period.end:
        return None
    start, end = period.start, period.end
    start_month, end_month = start.strftime("%Y-%m"), end.strftime("%Y-%m")
    if start.day == 1 and end == month_bounds(end_month)[1]:
        n = (end.year * 12 + end.month) - (start.year * 12 + start.month) + 1
        prev = DateRange(month_bounds(shift_month(start_month, -n))[0], month_bounds(shift_month(end_month, -n))[1])
    else:
        length = (end - start).days + 1
        prev_end = start - timedelta(days=1)
        prev = DateRange(prev_end - timedelta(days=length - 1), prev_end)
    return prev if prev.start >= info.first_order_date else None


def kpis_with_comparison(db: Database, info: DatasetInfo, period: DateRange = ALL_TIME) -> dict:
    current = sales_metrics(db, period)
    prev_period = previous_period(period, info)
    previous = sales_metrics(db, prev_period) if prev_period else None
    keys = ("revenue", "orders", "average_order_value", "customers")
    return {
        "current": current,
        "previous": previous,
        "previous_period": prev_period.describe() if prev_period else None,
        "change_pct": {k: pct_change(current[k], previous[k]) for k in keys} if previous else None,
    }
