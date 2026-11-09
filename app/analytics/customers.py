"""Customer analytics: totals, new/returning/repeat, inactive and top customers.

Definitions (completed orders only):
- new customer in a period: first-ever purchase falls inside the period
- returning customer: purchased in the period, first purchase was before it
- repeat customer: 2+ purchases inside the period
- inactive customer: has purchased before, but not within ``inactive_days`` of the dataset's latest date
"""

from __future__ import annotations

from datetime import date, timedelta

from app.analytics.periods import ALL_TIME, DateRange
from app.db import Database
from app.errors import InvalidInputError
from app.formatting import r2, share


def _validate_days(days: int) -> int:
    if not 7 <= int(days) <= 730:
        raise InvalidInputError("inactive_days must be between 7 and 730.")
    return int(days)


def customer_metrics(db: Database, as_of: date, period: DateRange = ALL_TIME, inactive_days: int = 90) -> dict:
    inactive_days = _validate_days(inactive_days)
    start, end = period.params()
    cutoff = (as_of - timedelta(days=inactive_days)).isoformat()
    row = db.query_one(
        """
        WITH firsts AS (
            SELECT customer_id, MIN(order_date) AS first_order_date
            FROM order_totals WHERE status = 'completed' GROUP BY customer_id
        ),
        in_period AS (
            SELECT customer_id, COUNT(*) AS orders, SUM(order_revenue) AS revenue
            FROM order_totals
            WHERE status = 'completed' AND order_date BETWEEN ? AND ?
            GROUP BY customer_id
        )
        SELECT
            (SELECT COUNT(*) FROM customers)                                        AS registered,
            (SELECT COUNT(*) FROM firsts)                                           AS ever_purchased,
            COUNT(p.customer_id)                                                    AS active,
            COALESCE(SUM(f.first_order_date BETWEEN ? AND ?), 0)                    AS new_customers,
            COALESCE(SUM(f.first_order_date < ?), 0)                                AS returning_customers,
            COALESCE(SUM(p.orders >= 2), 0)                                         AS repeat_customers,
            COALESCE(SUM(CASE WHEN p.orders >= 2 THEN p.revenue END), 0)            AS repeat_revenue,
            COALESCE(SUM(p.revenue), 0)                                             AS revenue,
            COALESCE(SUM(p.orders), 0)                                              AS orders
        FROM in_period p JOIN firsts f ON f.customer_id = p.customer_id
        """,
        (start, end, start, end, start),
    )
    inactive = db.query_one(
        """
        SELECT COUNT(*) AS n, COALESCE(SUM(revenue), 0) AS lifetime_revenue
        FROM customer_summary WHERE orders > 0 AND last_order_date < ?
        """,
        (cutoff,),
    )
    active = row["active"]
    return {
        "period": period.describe(),
        "total_registered_customers": row["registered"],
        "customers_who_ever_purchased": row["ever_purchased"],
        "never_purchased_customers": row["registered"] - row["ever_purchased"],
        "active_customers": active,
        "new_customers": row["new_customers"],
        "returning_customers": row["returning_customers"],
        "repeat_customers": row["repeat_customers"],
        "repeat_customer_rate_pct": share(row["repeat_customers"], active),
        "repeat_customer_revenue_share_pct": share(row["repeat_revenue"], row["revenue"]),
        "average_orders_per_customer": r2(row["orders"] / active) if active else None,
        "average_revenue_per_customer": r2(row["revenue"] / active) if active else None,
        "inactive_days_threshold": inactive_days,
        "inactive_customers": inactive["n"],
        "inactive_customers_lifetime_revenue": r2(inactive["lifetime_revenue"]),
        "as_of_date": as_of.isoformat(),
    }


def top_customers(db: Database, limit: int = 10, period: DateRange = ALL_TIME) -> list[dict]:
    if not 1 <= int(limit) <= 100:
        raise InvalidInputError("limit must be between 1 and 100.")
    rows = db.query(
        """
        SELECT ot.customer_id,
               c.first_name || ' ' || c.last_name AS customer_name,
               c.region,
               COUNT(*)              AS orders,
               SUM(ot.order_revenue) AS revenue,
               MAX(ot.order_date)    AS last_order_date
        FROM order_totals ot JOIN customers c ON c.customer_id = ot.customer_id
        WHERE ot.status = 'completed' AND ot.order_date BETWEEN ? AND ?
        GROUP BY ot.customer_id
        ORDER BY revenue DESC, ot.customer_id
        LIMIT ?
        """,
        (*period.params(), int(limit)),
    )
    return [{"rank": i + 1, **r, "revenue": r2(r["revenue"])} for i, r in enumerate(rows)]


def inactive_customers(db: Database, as_of: date, days: int = 90, limit: int = 20) -> dict:
    days = _validate_days(days)
    if not 1 <= int(limit) <= 200:
        raise InvalidInputError("limit must be between 1 and 200.")
    cutoff = (as_of - timedelta(days=days)).isoformat()
    rows = db.query(
        """
        SELECT customer_id, customer_name, email, region, last_order_date,
               orders AS lifetime_orders, revenue AS lifetime_revenue,
               CAST(julianday(?) - julianday(last_order_date) AS INTEGER) AS days_since_last_order
        FROM customer_summary
        WHERE orders > 0 AND last_order_date < ?
        ORDER BY revenue DESC, customer_id
        LIMIT ?
        """,
        (as_of.isoformat(), cutoff, int(limit)),
    )
    totals = db.query_one(
        """
        SELECT COUNT(*) AS n, COALESCE(SUM(revenue), 0) AS revenue,
               (SELECT COUNT(*) FROM customer_summary WHERE orders > 0) AS buyers,
               (SELECT COUNT(*) FROM customer_summary WHERE orders = 0) AS never
        FROM customer_summary WHERE orders > 0 AND last_order_date < ?
        """,
        (cutoff,),
    )
    return {
        "inactive_days_threshold": days,
        "as_of_date": as_of.isoformat(),
        "no_purchase_since": cutoff,
        "inactive_customer_count": totals["n"],
        "inactive_share_of_buyers_pct": share(totals["n"], totals["buyers"]),
        "inactive_lifetime_revenue": r2(totals["revenue"]),
        "never_purchased_customers": totals["never"],
        "customers": [{**r, "lifetime_revenue": r2(r["lifetime_revenue"])} for r in rows],
        "sorted_by": "lifetime_revenue (most valuable lapsed customers first)",
    }


def customer_trend(db: Database, period: DateRange = ALL_TIME) -> list[dict]:
    """Per month: active, new and returning customers plus the revenue each group generated."""
    start, end = period.params()
    rows = db.query(
        """
        WITH firsts AS (
            SELECT customer_id, substr(MIN(order_date), 1, 7) AS first_month
            FROM order_totals WHERE status = 'completed' GROUP BY customer_id
        )
        SELECT ot.month,
               COUNT(DISTINCT ot.customer_id) AS active_customers,
               COUNT(DISTINCT CASE WHEN f.first_month = ot.month THEN ot.customer_id END) AS new_customers,
               COUNT(DISTINCT CASE WHEN f.first_month < ot.month THEN ot.customer_id END) AS returning_customers,
               SUM(CASE WHEN f.first_month = ot.month THEN ot.order_revenue ELSE 0 END) AS new_customer_revenue,
               SUM(CASE WHEN f.first_month < ot.month THEN ot.order_revenue ELSE 0 END) AS returning_customer_revenue
        FROM order_totals ot JOIN firsts f ON f.customer_id = ot.customer_id
        WHERE ot.status = 'completed' AND ot.order_date BETWEEN ? AND ?
        GROUP BY ot.month ORDER BY ot.month
        """,
        (start, end),
    )
    return [
        {**r, "new_customer_revenue": r2(r["new_customer_revenue"]),
         "returning_customer_revenue": r2(r["returning_customer_revenue"])}
        for r in rows
    ]
