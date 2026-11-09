"""Consistent human-readable formatting for numbers in answers and insights."""

from __future__ import annotations

from datetime import datetime


def r2(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def pct_change(current: float | None, previous: float | None) -> float | None:
    """Percent change rounded to one decimal; ``None`` when the base is zero/missing."""
    if current is None or previous is None or previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 1)


def share(part: float, total: float) -> float | None:
    return None if not total else round(part / total * 100, 1)


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(value)
    return f"{sign}${v:,.0f}" if v >= 1000 else f"{sign}${v:,.2f}"


def signed_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return ("+" if value >= 0 else "") + money(value)


def pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def month_label(month: str) -> str:
    return datetime.strptime(month, "%Y-%m").strftime("%B %Y")
