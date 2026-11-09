"""Multi-step diagnostic: *why* did revenue change between two months?

The analysis runs a fixed sequence of investigations and returns both the raw
evidence and plain-language findings built only from that evidence:

1. headline KPIs for both months (revenue, orders, AOV, units, customers, cancellations)
2. decomposition of the revenue change into order-volume and order-value effects
3. category contribution to the change
4. product-level movers, flagging products whose sales nearly stopped
5. customer activity: new vs returning buyers and their revenue
6. regional contribution
"""

from __future__ import annotations

from app.analytics.customers import customer_trend
from app.analytics.periods import DatasetInfo, DateRange, month_bounds, parse_month, previous_month
from app.analytics.sales import revenue_breakdown, sales_metrics
from app.db import Database
from app.errors import InvalidInputError
from app.formatting import money, month_label, pct, pct_change, r2, share, signed_money

FLAT_THRESHOLD_PCT = 1.0
MATERIAL_SHARE_PCT = 15.0


def _kpis(db: Database, month: str) -> dict:
    m = sales_metrics(db, DateRange.for_month(month))
    return {
        "revenue": m["revenue"],
        "orders": m["orders"],
        "average_order_value": m["average_order_value"],
        "units_sold": m["units_sold"],
        "active_customers": m["customers"],
        "cancellation_rate_pct": m["cancellation_rate_pct"],
        "discounts_given": m["discounts_given"],
    }


def _compare(current: dict, previous: dict) -> dict:
    out = {}
    for key in current:
        c, p = current[key], previous[key]
        out[key] = {"previous": p, "current": c,
                    "change": r2(c - p) if c is not None and p is not None else None,
                    "change_pct": pct_change(c, p)}
    return out


def _dimension_change(db: Database, dimension: str, month: str, compare_to: str, total_change: float) -> list[dict]:
    cur = {r[dimension]: r["revenue"] for r in revenue_breakdown(db, dimension, DateRange.for_month(month))}
    prev = {r[dimension]: r["revenue"] for r in revenue_breakdown(db, dimension, DateRange.for_month(compare_to))}
    rows = []
    for key in sorted(set(cur) | set(prev)):
        c, p = cur.get(key, 0.0), prev.get(key, 0.0)
        rows.append({
            dimension: key,
            "revenue_previous": r2(p),
            "revenue_current": r2(c),
            "change": r2(c - p),
            "change_pct": pct_change(c, p),
            "share_of_total_change_pct": share(c - p, total_change) if total_change else None,
        })
    rows.sort(key=lambda r: r["change"], reverse=total_change > 0)
    return rows


def _product_changes(db: Database, month: str, compare_to: str) -> list[dict]:
    rows = db.query(
        """
        SELECT p.product_id, p.product_name, p.category,
               COALESCE(SUM(CASE WHEN s.month = :cur THEN s.line_revenue END), 0)  AS revenue_current,
               COALESCE(SUM(CASE WHEN s.month = :prev THEN s.line_revenue END), 0) AS revenue_previous,
               COALESCE(SUM(CASE WHEN s.month = :cur THEN s.quantity END), 0)      AS units_current,
               COALESCE(SUM(CASE WHEN s.month = :prev THEN s.quantity END), 0)     AS units_previous
        FROM products p
        LEFT JOIN sales s ON s.product_id = p.product_id AND s.month IN (:cur, :prev)
        GROUP BY p.product_id
        """,
        {"cur": month, "prev": compare_to},
    )
    for r in rows:
        r["change"] = r2(r["revenue_current"] - r["revenue_previous"])
        r["change_pct"] = pct_change(r["revenue_current"], r["revenue_previous"])
        r["revenue_current"] = r2(r["revenue_current"])
        r["revenue_previous"] = r2(r["revenue_previous"])
    return rows


def _customer_activity(db: Database, month: str, compare_to: str) -> dict:
    first = min(month_bounds(month)[0], month_bounds(compare_to)[0])
    last = max(month_bounds(month)[1], month_bounds(compare_to)[1])
    trend = {r["month"]: r for r in customer_trend(db, DateRange(first, last))}
    empty = {"active_customers": 0, "new_customers": 0, "returning_customers": 0,
             "new_customer_revenue": 0.0, "returning_customer_revenue": 0.0}
    cur, prev = trend.get(month, empty), trend.get(compare_to, empty)
    return {
        key: {"previous": prev[key], "current": cur[key], "change": r2(cur[key] - prev[key]),
              "change_pct": pct_change(cur[key], prev[key])}
        for key in empty
    }


def analyze_revenue_change(
    db: Database, info: DatasetInfo, month: str | None = None, compare_to: str | None = None
) -> dict:
    month = info.require_month(parse_month(month)) if month else info.latest_full_month
    compare_to = info.require_month(parse_month(compare_to)) if compare_to else previous_month(month)
    if compare_to == month:
        raise InvalidInputError("month and compare_to must be different months.")
    info.require_month(compare_to)

    steps: list[str] = []
    cur, prev = _kpis(db, month), _kpis(db, compare_to)
    headline = _compare(cur, prev)
    steps.append("Compared headline KPIs (revenue, orders, AOV, units, customers, cancellations) for both months")

    delta = cur["revenue"] - prev["revenue"]
    change_pct = pct_change(cur["revenue"], prev["revenue"])
    if change_pct is None or abs(change_pct) < FLAT_THRESHOLD_PCT:
        direction = "flat"
    else:
        direction = "decline" if delta < 0 else "increase"

    decomposition = None
    if prev["orders"] and cur["orders"]:
        volume = (cur["orders"] - prev["orders"]) * prev["average_order_value"]
        value = (cur["average_order_value"] - prev["average_order_value"]) * cur["orders"]
        decomposition = {
            "revenue_change": r2(delta),
            "order_volume_effect": r2(volume),
            "order_value_effect": r2(value),
            "dominant_driver": "order volume" if abs(volume) >= abs(value) else "average order value",
            "method": "volume = (orders_cur - orders_prev) * AOV_prev; value = (AOV_cur - AOV_prev) * orders_cur",
        }
        steps.append("Decomposed the revenue change into order-volume and order-value (AOV) effects")

    categories = _dimension_change(db, "category", month, compare_to, delta)
    steps.append("Measured each category's contribution to the change")

    products = _product_changes(db, month, compare_to)
    decliners = sorted((p for p in products if p["change"] < 0), key=lambda p: p["change"])[:5]
    gainers = sorted((p for p in products if p["change"] > 0), key=lambda p: -p["change"])[:3]
    stalled = [p for p in products if p["units_previous"] >= 10 and p["units_current"] <= 0.25 * p["units_previous"]]
    steps.append("Ranked product-level movers and flagged products whose sales nearly stopped")

    customers = _customer_activity(db, month, compare_to)
    steps.append("Compared new vs returning customer activity and revenue")

    regions = _dimension_change(db, "region", month, compare_to, delta)
    steps.append("Measured each region's contribution to the change")

    result = {
        "month": month,
        "compare_to": compare_to,
        "direction": direction,
        "revenue_change": r2(delta),
        "revenue_change_pct": change_pct,
        "headline": headline,
        "decomposition": decomposition,
        "categories": categories,
        "top_declining_products": decliners,
        "top_growing_products": gainers,
        "products_with_sales_nearly_stopped": stalled,
        "customers": customers,
        "regions": regions,
        "steps": steps,
    }
    result["findings"] = build_findings(result)
    return result


def build_findings(r: dict) -> list[str]:
    """Plain-language findings; every number comes from the analysis dict."""
    cur_l, prev_l = month_label(r["month"]), month_label(r["compare_to"])
    h = r["headline"]
    rev = h["revenue"]
    findings = []
    if r["direction"] == "decline":
        findings.append(
            f"Revenue fell {pct(abs(r['revenue_change_pct']))} from {money(rev['previous'])} in {prev_l} "
            f"to {money(rev['current'])} in {cur_l} ({signed_money(r['revenue_change'])})."
        )
    elif r["direction"] == "increase":
        findings.append(
            f"Revenue did not decline: it rose {pct(r['revenue_change_pct'])} from {money(rev['previous'])} "
            f"in {prev_l} to {money(rev['current'])} in {cur_l} ({signed_money(r['revenue_change'])})."
        )
    else:
        findings.append(
            f"Revenue was essentially flat: {money(rev['previous'])} in {prev_l} vs {money(rev['current'])} in {cur_l}."
        )
        return findings

    o, a = h["orders"], h["average_order_value"]
    d = r["decomposition"]
    if d:
        findings.append(
            f"Orders went from {o['previous']:,} to {o['current']:,} ({pct(o['change_pct'], signed=True)}) and "
            f"average order value from {money(a['previous'])} to {money(a['current'])} ({pct(a['change_pct'], signed=True)}). "
            f"Order volume explains {signed_money(d['order_volume_effect'])} of the change and order value "
            f"{signed_money(d['order_value_effect'])}."
        )

    same_direction = [c for c in r["categories"]
                      if c["share_of_total_change_pct"] and c["share_of_total_change_pct"] >= MATERIAL_SHARE_PCT]
    for c in same_direction[:2]:
        verb = "fell" if c["change"] < 0 else "rose"
        findings.append(
            f"{c['category']} revenue {verb} {pct(abs(c['change_pct'] or 0))} ({money(c['revenue_previous'])} to "
            f"{money(c['revenue_current'])}), accounting for {pct(c['share_of_total_change_pct'])} of the total change."
        )

    stalled = r["products_with_sales_nearly_stopped"]
    if stalled:
        parts = ", ".join(f"{p['product_name']} ({p['units_previous']} to {p['units_current']} units)" for p in stalled[:4])
        findings.append(f"Sales nearly stopped for {parts}, which suggests a stock-out or listing problem.")
    elif r["direction"] == "decline" and r["top_declining_products"]:
        parts = ", ".join(f"{p['product_name']} ({signed_money(p['change'])})" for p in r["top_declining_products"][:3])
        findings.append(f"The largest product-level declines were {parts}.")

    nc, rc = r["customers"]["new_customers"], r["customers"]["returning_customers"]
    if nc["change_pct"] is not None and abs(nc["change_pct"]) >= 10:
        verb = "fell" if nc["change"] < 0 else "rose"
        findings.append(
            f"New customers {verb} from {nc['previous']:,} to {nc['current']:,} ({pct(nc['change_pct'], signed=True)}), "
            f"while returning customers went from {rc['previous']:,} to {rc['current']:,} ({pct(rc['change_pct'], signed=True)})."
        )

    cr = h["cancellation_rate_pct"]
    if cr["previous"] is not None and cr["current"] is not None and abs(cr["current"] - cr["previous"]) >= 2:
        verb = "rose" if cr["current"] > cr["previous"] else "fell"
        findings.append(
            f"The order cancellation rate {verb} from {pct(cr['previous'])} to {pct(cr['current'])}, "
            f"so fewer placed orders turned into revenue."
        )
    return findings
