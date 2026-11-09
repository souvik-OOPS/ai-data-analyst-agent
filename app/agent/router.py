"""Deterministic rule-based analyst.

Used when no LLM is configured, when the LLM API fails, and as a safety net when
an LLM answer cannot be verified. It maps common business questions to the same
tool registry the LLM uses, then formats answers from the tool outputs only.
It understands fewer phrasings than the LLM, by design: anything it cannot map
confidently is reported as unsupported rather than guessed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date

from app.agent.grounding import Evidence, check_grounding
from app.agent.models import AgentAnswer, ToolCallRecord
from app.agent.tools import ToolRegistry, ToolResult
from app.analytics.periods import DatasetInfo, month_bounds, shift_month
from app.formatting import money, month_label, num, pct

EXAMPLE_QUESTIONS = [
    "What were the top 5 products by revenue?",
    "Show monthly revenue.",
    "Which customers have not purchased recently?",
    "Which category generated the most revenue?",
    "Why did revenue decline last month?",
    "Show me monthly revenue as a chart.",
]

UNSUPPORTED_TOPICS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(profit|margins?|cogs|cost of goods|gross profit|net income|ebitda|unit costs?)\b"),
     "The dataset contains selling prices and discounts but no product costs, so profit and margins cannot be "
     "calculated."),
    (re.compile(r"\b(marketing|ad spend|advertising|campaigns?|roas|cac|acquisition cost|traffic|sessions|"
                r"page ?views|conversion rate|visits)\b"),
     "There is no marketing, campaign or website-traffic data in the dataset."),
    (re.compile(r"\b(inventory|stock levels?|warehouse|suppliers?|shipping times?|delivery times?)\b"),
     "There is no inventory, supplier or logistics data in the dataset (stock-outs can only be inferred from "
     "sales patterns)."),
    (re.compile(r"\b(forecast|predict|prediction|projection|next (month|quarter|year)|will .{0,40}\b(be|reach))\b"),
     "Forecasting is not supported: the analyst reports on historical data only."),
    (re.compile(r"\b(employees?|salary|salaries|staff|payroll)\b"), "There is no HR or staffing data in the dataset."),
    (re.compile(r"\b(weather|stock price|share price|bitcoin|crypto|news|election|recipe|joke|poem)\b"),
     "That question is unrelated to the company's sales data."),
]

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s+(\d{4}))?"
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20}


@dataclass
class Intent:
    name: str
    params: dict = field(default_factory=dict)
    period_label: str | None = None
    reason: str | None = None


def _has(q: str, pattern: str) -> bool:
    return re.search(pattern, q) is not None


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


class RuleBasedAnalyst:
    def __init__(self, registry: ToolRegistry, info: DatasetInfo):
        self.registry = registry
        self.info = info

    # ------------------------------------------------------------------ parsing

    def _period(self, q: str) -> tuple[dict, str | None]:
        info = self.info
        latest = info.latest_full_month

        def months_back(n: int) -> tuple[dict, str]:
            start = month_bounds(shift_month(latest, -(n - 1)))[0]
            return ({"start_date": start.isoformat(), "end_date": month_bounds(latest)[1].isoformat()},
                    f"{month_label(shift_month(latest, -(n - 1)))} to {month_label(latest)}")

        if _has(q, r"\b(last|previous|past) month\b"):
            first, last = month_bounds(latest)
            return {"start_date": first.isoformat(), "end_date": last.isoformat()}, month_label(latest)
        m = re.search(r"\b(?:last|past|previous) (\d{1,2}|" + "|".join(_NUMBER_WORDS) + r") months\b", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
            return months_back(max(1, min(n, len(info.months))))
        m = re.search(r"\b(?:last|past) (\d{1,3}) days\b", q)
        if m:
            r = info.relative_range(int(m.group(1)))
            return {"start_date": r.start.isoformat(), "end_date": r.end.isoformat()}, f"the last {m.group(1)} days"
        if _has(q, r"\b(last|past) (year|12 months)\b"):
            return months_back(12)
        if _has(q, r"\b(this year|ytd|year to date)\b"):
            start = date(info.as_of.year, 1, 1)
            return {"start_date": start.isoformat(), "end_date": info.as_of.isoformat()}, f"{info.as_of.year} to date"
        m = _MONTH_RE.search(q)
        if m and not (m.group(1) == "may" and not m.group(2)):
            name = next(k for k in _MONTHS if k.startswith(m.group(1)[:3]))
            candidates = [mo for mo in info.months if int(mo[5:]) == _MONTHS[name]
                          and (not m.group(2) or mo[:4] == m.group(2))]
            if candidates:
                first, last = month_bounds(candidates[-1])
                return {"start_date": first.isoformat(), "end_date": last.isoformat()}, month_label(candidates[-1])
        m = re.search(r"\b(?:in|for|during|of)\s+(20\d{2})\b", q)
        if m:
            year = int(m.group(1))
            return {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}, str(year)
        return {}, None

    def _month_param(self, q: str) -> str | None:
        m = _MONTH_RE.search(q)
        if m and not (m.group(1) == "may" and not m.group(2)):
            name = next(k for k in _MONTHS if k.startswith(m.group(1)[:3]))
            candidates = [mo for mo in self.info.months if int(mo[5:]) == _MONTHS[name]
                          and (not m.group(2) or mo[:4] == m.group(2))]
            return candidates[-1] if candidates else None
        return None

    @staticmethod
    def _limit(q: str, default: int) -> int:
        m = re.search(r"\b(?:top|bottom|best|worst|lowest|highest|least|first)\s+(\d{1,2}|"
                      + "|".join(_NUMBER_WORDS) + r")\b", q)
        if not m:
            m = re.search(r"\b(\d{1,2})\s+(?:products|customers|items)\b", q)
        if not m:
            return default
        value = m.group(1)
        return max(1, min(int(value) if value.isdigit() else _NUMBER_WORDS[value], 50))

    def detect(self, question: str) -> Intent:
        q = " ".join(question.lower().split())
        if not re.search(r"[a-z]{2,}", q):
            return Intent("invalid", reason="The question does not contain any words I can interpret.")
        for pattern, reason in UNSUPPORTED_TOPICS:
            if pattern.search(q):
                return Intent("unsupported", reason=reason)

        period, label = self._period(q)
        wants_chart = _has(q, r"\b(chart|graph|plot|visuali[sz]e|visual|draw)\b")
        revenue_words = r"\b(revenue|sales|income|turnover)\b"

        if (_has(q, r"\b(why|reason|explain|cause|caused|driver|drivers|drove|what happened)\b")
                and _has(q, revenue_words)) or _has(q, r"\brevenue (decline|drop|fall|dip|decrease)"):
            return Intent("revenue_change", {"month": self._month_param(q)} if self._month_param(q) else {})

        if wants_chart:
            chart_type = next((t for t in ("pie", "bar", "line") if _has(q, rf"\b{t}\b")), "auto")
            if _has(q, r"\bcategor"):
                chart = "revenue_by_category"
            elif _has(q, r"\bregion"):
                chart = "revenue_by_region"
            elif _has(q, r"\bchannel"):
                chart = "revenue_by_channel"
            elif _has(q, r"\bproducts?\b") and _has(q, r"\b(worst|bottom|low|lowest|least|under)"):
                chart = "bottom_products"
            elif _has(q, r"\bproducts?\b"):
                chart = "top_products"
            elif _has(q, r"\bcustomer"):
                chart = "customer_trend"
            elif _has(q, r"\b(aov|average order)"):
                chart = "aov_trend"
            elif _has(q, r"\borders?\b") and not _has(q, revenue_words):
                chart = "monthly_orders"
            else:
                chart = "monthly_revenue"
            if chart_type == "pie" and not chart.startswith("revenue_by_"):
                chart_type = "auto"
            params = {"chart": chart, "chart_type": chart_type, **period}
            if chart.endswith("_products"):
                params["limit"] = max(3, self._limit(q, 10))
            return Intent("chart", params, label)

        if _has(q, r"\b(not (purchased|bought|ordered)|haven'?t (purchased|bought|ordered)|inactive|lapsed|dormant|"
                   r"churn(ed)?|no (recent )?purchases?|stopped (buying|purchasing))\b"):
            m = re.search(r"\b(\d{1,3})\s+days\b", q)
            days = max(7, min(int(m.group(1)), 730)) if m else 90
            return Intent("inactive_customers", {"days": days, "limit": 10})

        if _has(q, r"\bcustomers?\b") and _has(q, r"\b(top|best|biggest|most valuable|highest|largest)\b"):
            return Intent("top_customers", {"limit": self._limit(q, 10), **period}, label)

        if _has(q, r"\b(new|repeat|returning|retention|loyal)\b.*\bcustomers?\b|\bhow many customers\b|"
                   r"\b(total|number of|count of) customers\b|\bcustomer (metrics|count|base)\b"):
            return Intent("customer_metrics", period, label)

        if _has(q, r"\b(products?|items?|skus?)\b"):
            if _has(q, r"\b(worst|bottom|low|lowest|least|under-?perform\w*|poor\w*|weak\w*)\b"):
                return Intent("products", {"direction": "bottom", "metric": "revenue",
                                           "limit": self._limit(q, 10), **period}, label)
            metric = "units" if _has(q, r"\b(units?|quantity|volume|selling|sold|sell|best-?sellers?)\b") \
                and not _has(q, r"\brevenue\b") else "revenue"
            return Intent("products", {"direction": "top", "metric": metric, "limit": self._limit(q, 10), **period},
                          label)

        for dim in ("category", "region", "channel"):
            stem = {"category": r"categor", "region": r"region", "channel": r"channel"}[dim]
            if _has(q, rf"\b{stem}"):
                return Intent("breakdown", {"dimension": dim, **period}, label)

        if _has(q, r"\b(monthly|per month|by month|each month|month by month|over time|trend)\b"):
            return Intent("monthly", period, label)

        if _has(q, revenue_words) or _has(q, r"\b(orders?|aov|average order value|kpis?|how much)\b"):
            return Intent("sales_metrics", period, label)

        if _has(q, r"\b(what data|which data|date range|time period|coverage|what can you)\b"):
            return Intent("dataset_info")

        return Intent("unknown", reason="I could not map this question to one of the analyses I support.")

    # ------------------------------------------------------------------ answering

    def answer(self, question: str) -> AgentAnswer:
        started = time.perf_counter()
        intent = self.detect(question)
        results: list[ToolResult] = []

        def run(tool: str, args: dict) -> ToolResult:
            result = self.registry.execute(tool, args)
            results.append(result)
            return result

        status = "answered"
        if intent.name in ("invalid", "unsupported", "unknown"):
            status = "unsupported"
            text = self._unsupported_text(intent)
        else:
            text = getattr(self, f"_answer_{intent.name}")(intent, run)
            failed = [r for r in results if not r.ok]
            if failed and not any(r.ok for r in results):
                status = "error"
                text = "I could not compute this: " + "; ".join(r.error or "unknown error" for r in failed)

        evidence = Evidence([r.output for r in results if r.ok] + [self.info.to_dict()])
        return AgentAnswer(
            question=question,
            answer=text,
            status=status,
            engine="rule_based",
            tool_calls=[ToolCallRecord(tool=r.name, input=r.input, ok=r.ok, duration_ms=r.duration_ms,
                                       output_preview=r.preview() if r.ok else None, error=r.error)
                        for r in results],
            charts=[r.chart for r in results if r.chart],
            grounding=check_grounding(text, evidence, question) if status == "answered" else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _unsupported_text(self, intent: Intent) -> str:
        examples = "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)
        return (f"**I can't answer that from this dataset.** {intent.reason}\n\n"
                f"I can analyse revenue, orders, average order value, products, categories, regions, channels and "
                f"customers from {self.info.first_order_date.isoformat()} to {self.info.last_order_date.isoformat()}. "
                f"For example:\n{examples}")

    @staticmethod
    def _scope(intent: Intent) -> str:
        return f" ({intent.period_label})" if intent.period_label else " (all data)"

    def _answer_sales_metrics(self, intent: Intent, run) -> str:
        r = run("get_sales_metrics", intent.params)
        if not r.ok:
            return ""
        o = r.output
        if not o["orders"]:
            return f"There were no completed orders{self._scope(intent)}."
        return (f"**Revenue{self._scope(intent)} was {money(o['revenue'])}** from {num(o['orders'])} completed orders, "
                f"an average order value of {money(o['average_order_value'])}.\n\n"
                f"- Purchasing customers: {num(o['customers'])}\n- Units sold: {num(o['units_sold'])}\n"
                f"- Discounts given: {money(o['discounts_given'])}\n"
                f"- Cancellation rate: {pct(o['cancellation_rate_pct'])} ({num(o['cancelled_orders'])} cancelled, "
                f"{num(o['returned_orders'])} returned orders excluded from revenue)")

    def _answer_monthly(self, intent: Intent, run) -> str:
        r = run("get_monthly_revenue", intent.params)
        if not r.ok:
            return ""
        months = r.output["months"]
        if not months:
            return f"There is no revenue data{self._scope(intent)}."
        best = max(months, key=lambda m: m["revenue"])
        worst = min(months, key=lambda m: m["revenue"])
        rows = [[month_label(m["month"]), money(m["revenue"]), num(m["orders"]), money(m["average_order_value"]),
                 pct(m["revenue_change_pct"], signed=True) if m["revenue_change_pct"] is not None else "-"]
                for m in months]
        latest = months[-1]
        return (f"**Monthly revenue{self._scope(intent)}.** The most recent month, {month_label(latest['month'])}, "
                f"brought in {money(latest['revenue'])}. The best month was {month_label(best['month'])} "
                f"({money(best['revenue'])}) and the weakest was {month_label(worst['month'])} "
                f"({money(worst['revenue'])}).\n\n"
                + _table(["Month", "Revenue", "Orders", "AOV", "MoM change"], rows))

    def _answer_breakdown(self, intent: Intent, run) -> str:
        r = run("get_revenue_breakdown", intent.params)
        if not r.ok:
            return ""
        dim = intent.params["dimension"]
        rows = r.output["rows"]
        if not rows:
            return f"There is no revenue data{self._scope(intent)}."
        top = rows[0]
        table = _table([dim.title(), "Revenue", "Share", "Orders"],
                       [[x[dim], money(x["revenue"]), pct(x["share_pct"]), num(x["orders"])] for x in rows])
        return (f"**{top[dim]} generated the most revenue{self._scope(intent)}: {money(top['revenue'])}**, "
                f"{pct(top['share_pct'])} of the total.\n\n{table}")

    def _answer_products(self, intent: Intent, run) -> str:
        r = run("get_top_products", intent.params)
        if not r.ok:
            return ""
        p = r.output
        products = p["products"]
        metric_label = "revenue" if p["metric"] == "revenue" else "units sold"
        rows = [[x["rank"], x["product_name"], x["category"], money(x["revenue"]), num(x["units_sold"]),
                 pct(x["revenue_share_pct"])] for x in products]
        table = _table(["#", "Product", "Category", "Revenue", "Units", "Share of revenue"], rows)
        if p["direction"] == "top":
            lead = products[0]
            headline = (f"**{lead['product_name']} is the top product by {metric_label}{self._scope(intent)}**, with "
                        f"{money(lead['revenue'])} from {num(lead['units_sold'])} units.")
            return f"{headline}\n\nTop {len(products)} products by {metric_label}:\n\n{table}"
        zero = [x["product_name"] for x in products if x["units_sold"] == 0]
        note = f"\n\n{', '.join(zero)} had no sales at all in this period." if zero else ""
        return (f"**Lowest-performing {len(products)} products by revenue{self._scope(intent)}:**\n\n{table}{note}")

    def _answer_customer_metrics(self, intent: Intent, run) -> str:
        r = run("get_customer_metrics", intent.params)
        if not r.ok:
            return ""
        c = r.output
        return (f"**Customer overview{self._scope(intent)}:** {num(c['active_customers'])} customers purchased, "
                f"of whom {num(c['new_customers'])} were new and {num(c['returning_customers'])} were returning.\n\n"
                f"- Registered customers: {num(c['total_registered_customers'])} "
                f"({num(c['never_purchased_customers'])} have never purchased)\n"
                f"- Repeat customers (2+ orders): {num(c['repeat_customers'])} "
                f"({pct(c['repeat_customer_rate_pct'])} of buyers), generating "
                f"{pct(c['repeat_customer_revenue_share_pct'])} of revenue\n"
                f"- Average orders per customer: {num(c['average_orders_per_customer'])}; average revenue per "
                f"customer: {money(c['average_revenue_per_customer'])}\n"
                f"- Inactive customers (no purchase in {c['inactive_days_threshold']} days): "
                f"{num(c['inactive_customers'])}")

    def _answer_top_customers(self, intent: Intent, run) -> str:
        r = run("get_top_customers", intent.params)
        if not r.ok:
            return ""
        customers = r.output["customers"]
        if not customers:
            return f"No customers purchased{self._scope(intent)}."
        rows = [[c["rank"], c["customer_name"], c["customer_id"], c["region"], num(c["orders"]), money(c["revenue"]),
                 c["last_order_date"]] for c in customers]
        return (f"**Top {len(customers)} customers by revenue{self._scope(intent)}:**\n\n"
                + _table(["#", "Customer", "ID", "Region", "Orders", "Revenue", "Last order"], rows))

    def _answer_inactive_customers(self, intent: Intent, run) -> str:
        r = run("get_inactive_customers", intent.params)
        if not r.ok:
            return ""
        o = r.output
        rows = [[c["customer_name"], c["customer_id"], c["region"], c["last_order_date"],
                 num(c["days_since_last_order"]), num(c["lifetime_orders"]), money(c["lifetime_revenue"])]
                for c in o["customers"]]
        return (f"**{num(o['inactive_customer_count'])} customers have not purchased in the last "
                f"{o['inactive_days_threshold']} days** (no order since {o['no_purchase_since']}; the data runs to "
                f"{o['as_of_date']}). That is {pct(o['inactive_share_of_buyers_pct'])} of all buyers, with "
                f"{money(o['inactive_lifetime_revenue'])} in lifetime revenue between them. Another "
                f"{num(o['never_purchased_customers'])} registered customers have never purchased.\n\n"
                f"Most valuable lapsed customers:\n\n"
                + _table(["Customer", "ID", "Region", "Last order", "Days since", "Orders", "Lifetime revenue"], rows))

    def _answer_revenue_change(self, intent: Intent, run) -> str:
        r = run("analyze_revenue_change", intent.params)
        if not r.ok:
            return ""
        a = r.output
        start = month_bounds(shift_month(a["month"], -11))[0]
        if shift_month(a["month"], -11) < self.info.months[0]:
            start = self.info.first_order_date
        run("generate_chart", {"chart": "monthly_revenue", "start_date": start.isoformat(),
                               "end_date": month_bounds(a["month"])[1].isoformat()})
        findings = a["findings"]
        body = "\n".join(f"- {f}" for f in findings[1:])
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(a["steps"], start=1))
        title = {"decline": "Why revenue declined", "increase": "Revenue did not decline",
                 "flat": "Revenue was flat"}[a["direction"]]
        drivers = f"\n\n**What drove it:**\n{body}" if body else ""
        return (f"**{title}: {month_label(a['month'])} vs {month_label(a['compare_to'])}.** {findings[0]}"
                f"{drivers}\n\n**How this was investigated:**\n{steps}")

    def _answer_chart(self, intent: Intent, run) -> str:
        r = run("generate_chart", intent.params)
        if not r.ok:
            return ""
        spec = r.chart
        series = spec["series"][0]
        points = list(zip(series["x"], series["y"], strict=True))
        fmt = money if spec["value_format"] == "currency" else num
        if spec["chart"] in ("monthly_revenue", "monthly_orders", "aov_trend"):
            best = max(points, key=lambda p: p[1])
            summary = (f"It covers {month_label(points[0][0])} to {month_label(points[-1][0])}. The peak was "
                       f"{month_label(best[0])} at {fmt(best[1])}, and the latest month "
                       f"({month_label(points[-1][0])}) was {fmt(points[-1][1])}.")
        elif spec["chart"] == "customer_trend":
            returning = spec["series"][1]["y"][-1]
            summary = (f"In {month_label(points[-1][0])} there were {num(points[-1][1])} new and {num(returning)} "
                       f"returning customers.")
        else:
            summary = f"The leader is {points[0][0]} with {fmt(points[0][1])}."
        return f"**Here is the {spec['title'].lower()} chart{self._scope(intent)}.** {summary}"

    def _answer_dataset_info(self, intent: Intent, run) -> str:
        r = run("get_dataset_info", {})
        if not r.ok:
            return ""
        o = r.output
        return (f"The dataset covers **{num(o['orders'])} orders from {o['first_order_date']} to "
                f"{o['last_order_date']}**, {num(o['customers'])} customers and {num(o['products'])} products across "
                f"{len(o['categories'])} categories ({', '.join(o['categories'])}). Relative dates such as "
                f"\"last month\" refer to {month_label(o['latest_full_month'])}.")
