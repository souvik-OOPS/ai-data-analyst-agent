"""Product performance: top sellers, highest revenue and low performers."""

from __future__ import annotations

from typing import Literal

from app.analytics.periods import ALL_TIME, DateRange
from app.db import Database
from app.errors import InvalidInputError
from app.formatting import r2, share

ProductMetric = Literal["revenue", "units"]
Direction = Literal["top", "bottom"]

_ORDER_BY = {
    ("revenue", "top"): "revenue DESC, units DESC, p.product_name",
    ("revenue", "bottom"): "revenue ASC, units ASC, p.product_name",
    ("units", "top"): "units DESC, revenue DESC, p.product_name",
    ("units", "bottom"): "units ASC, revenue ASC, p.product_name",
}


def resolve_category(db: Database, category: str | None) -> str | None:
    """Case-insensitive match against known categories; raises with the valid options."""
    if category is None or not str(category).strip():
        return None
    known = [r["category"] for r in db.query("SELECT DISTINCT category FROM products ORDER BY category")]
    for name in known:
        if name.lower() == str(category).strip().lower():
            return name
    raise InvalidInputError(f"Unknown category '{category}'. Valid categories: {', '.join(known)}.")


def product_performance(
    db: Database,
    metric: ProductMetric = "revenue",
    direction: Direction = "top",
    limit: int = 10,
    category: str | None = None,
    period: DateRange = ALL_TIME,
) -> dict:
    """Rank products. Products with no sales in the period are included (as zero) so
    'bottom' rankings surface items that did not sell at all."""
    if (metric, direction) not in _ORDER_BY:
        raise InvalidInputError("metric must be 'revenue' or 'units'; direction must be 'top' or 'bottom'.")
    if not 1 <= int(limit) <= 100:
        raise InvalidInputError("limit must be between 1 and 100.")
    category = resolve_category(db, category)
    start, end = period.params()
    rows = db.query(
        f"""
        SELECT p.product_id, p.product_name, p.category, p.list_price,
               COALESCE(SUM(s.quantity), 0)     AS units,
               COALESCE(SUM(s.line_revenue), 0) AS revenue,
               COUNT(DISTINCT s.order_id)       AS orders
        FROM products p
        LEFT JOIN sales s ON s.product_id = p.product_id AND s.order_date BETWEEN ? AND ?
        WHERE (? IS NULL OR p.category = ?)
        GROUP BY p.product_id
        ORDER BY {_ORDER_BY[(metric, direction)]}
        LIMIT ?
        """,
        (start, end, category, category, int(limit)),
    )
    total = db.scalar(
        "SELECT COALESCE(SUM(line_revenue), 0) FROM sales WHERE order_date BETWEEN ? AND ? AND (? IS NULL OR category = ?)",
        (start, end, category, category),
    )
    return {
        "metric": metric,
        "direction": direction,
        "category": category,
        "period": period.describe(),
        "products": [
            {
                "rank": i + 1,
                "product_id": r["product_id"],
                "product_name": r["product_name"],
                "category": r["category"],
                "revenue": r2(r["revenue"]),
                "units_sold": r["units"],
                "orders": r["orders"],
                "revenue_share_pct": share(r["revenue"], total),
            }
            for i, r in enumerate(rows)
        ],
        "total_revenue_in_scope": r2(total),
    }
