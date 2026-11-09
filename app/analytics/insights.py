"""Data-backed business insights.

``compute_insights`` produces a fixed set of insights, each with the evidence
(numbers) it was built from. The text is template-based so it is correct by
construction. The optional LLM writer (``app.agent.insight_writer``) can rephrase
these, and its output is verified against the same evidence.
"""

from __future__ import annotations

from app.analytics.customers import customer_metrics
from app.analytics.diagnostics import analyze_revenue_change
from app.analytics.periods import DatasetInfo, DateRange, shift_month
from app.analytics.products import product_performance
from app.analytics.sales import monthly_sales, revenue_breakdown
from app.db import Database
from app.formatting import money, month_label, pct, pct_change


def compute_insights(db: Database, info: DatasetInfo) -> list[dict]:
    insights: list[dict] = []
    months = {m["month"]: m for m in monthly_sales(db)}
    latest = info.latest_full_month

    # 1. Latest month vs previous month, with the main driver.
    if shift_month(latest, -1) in months:
        diag = analyze_revenue_change(db, info, latest)
        driver = next((c for c in diag["categories"] if (c["share_of_total_change_pct"] or 0) >= 15), None)
        text = diag["findings"][0]
        evidence = {"revenue_current": diag["headline"]["revenue"]["current"],
                    "revenue_previous": diag["headline"]["revenue"]["previous"],
                    "revenue_change": diag["revenue_change"], "revenue_change_pct": diag["revenue_change_pct"]}
        if driver and diag["direction"] != "flat":
            text += (f" {driver['category']} accounted for {pct(driver['share_of_total_change_pct'])} of the change "
                     f"({money(driver['change'])}).")
            evidence.update({"driver_category": driver["category"], "driver_change": driver["change"],
                             "driver_share_pct": driver["share_of_total_change_pct"]})
        insights.append({"id": "latest_month", "title": f"{month_label(latest)} vs previous month",
                         "sentiment": {"decline": "negative", "increase": "positive"}.get(diag["direction"], "neutral"),
                         "text": text, "evidence": evidence})

    # 2. Year over year for the latest month.
    year_ago = shift_month(latest, -12)
    if year_ago in months and latest in months:
        cur, prev = months[latest]["revenue"], months[year_ago]["revenue"]
        change = pct_change(cur, prev)
        word = "up" if (change or 0) >= 0 else "down"
        insights.append({
            "id": "year_over_year", "title": "Year-over-year",
            "sentiment": "positive" if (change or 0) > 0 else "negative",
            "text": f"{month_label(latest)} revenue was {money(cur)}, {word} {pct(abs(change or 0))} from "
                    f"{month_label(year_ago)} ({money(prev)}).",
            "evidence": {"revenue_current": cur, "revenue_year_ago": prev, "change_pct": change},
        })

    # 3. Peak month.
    if months:
        peak = max(months.values(), key=lambda m: m["revenue"])
        holiday = " (holiday season)" if peak["month"][5:] in ("11", "12") else ""
        insights.append({
            "id": "peak_month", "title": "Peak month", "sentiment": "neutral",
            "text": f"The best month was {month_label(peak['month'])}{holiday} with {money(peak['revenue'])} "
                    f"from {peak['orders']:,} orders.",
            "evidence": {"month": peak["month"], "revenue": peak["revenue"], "orders": peak["orders"]},
        })

    # 4. Category concentration.
    cats = revenue_breakdown(db, "category")
    if cats:
        top = cats[0]
        insights.append({
            "id": "top_category", "title": "Largest category", "sentiment": "neutral",
            "text": f"{top['category']} is the largest category with {money(top['revenue'])} "
                    f"({pct(top['share_pct'])} of all revenue).",
            "evidence": {"category": top["category"], "revenue": top["revenue"], "share_pct": top["share_pct"]},
        })

    # 5. Product concentration.
    top5 = product_performance(db, "revenue", "top", 5)
    share5 = round(sum(p["revenue_share_pct"] or 0 for p in top5["products"]), 1)
    insights.append({
        "id": "product_concentration", "title": "Product concentration", "sentiment": "neutral",
        "text": f"The top 5 products generate {pct(share5)} of revenue, led by "
                f"{top5['products'][0]['product_name']} ({money(top5['products'][0]['revenue'])}).",
        "evidence": {"top5_share_pct": share5, "top_product": top5["products"][0]["product_name"],
                     "top_product_revenue": top5["products"][0]["revenue"]},
    })

    # 6. Customer loyalty and churn risk.
    cm = customer_metrics(db, info.as_of, DateRange(), 90)
    insights.append({
        "id": "repeat_customers", "title": "Repeat customers", "sentiment": "positive",
        "text": f"{pct(cm['repeat_customer_rate_pct'])} of buyers have purchased more than once, and they generate "
                f"{pct(cm['repeat_customer_revenue_share_pct'])} of revenue.",
        "evidence": {"repeat_customer_rate_pct": cm["repeat_customer_rate_pct"],
                     "repeat_customer_revenue_share_pct": cm["repeat_customer_revenue_share_pct"]},
    })
    insights.append({
        "id": "inactive_customers", "title": "Win-back opportunity", "sentiment": "negative",
        "text": f"{cm['inactive_customers']:,} past buyers have not purchased in 90+ days; together they have spent "
                f"{money(cm['inactive_customers_lifetime_revenue'])} to date, which makes them a win-back target.",
        "evidence": {"inactive_customers": cm["inactive_customers"],
                     "inactive_customers_lifetime_revenue": cm["inactive_customers_lifetime_revenue"]},
    })
    return insights
