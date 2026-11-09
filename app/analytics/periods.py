"""Date ranges, month helpers and dataset coverage.

Relative periods such as "last month" are anchored to the dataset's latest
order date, not the wall clock: the sample data is static, so "last month"
means the most recent complete month present in the data.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.db import Database
from app.errors import DataNotFoundError, InvalidInputError

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
MIN_DATE = "0001-01-01"
MAX_DATE = "9999-12-31"


@dataclass(frozen=True)
class DateRange:
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.start and self.end and self.start > self.end:
            raise InvalidInputError(f"start_date ({self.start}) must be on or before end_date ({self.end}).")

    @classmethod
    def for_month(cls, month: str) -> DateRange:
        first, last = month_bounds(month)
        return cls(first, last)

    def params(self) -> tuple[str, str]:
        return (self.start.isoformat() if self.start else MIN_DATE, self.end.isoformat() if self.end else MAX_DATE)

    def describe(self) -> dict[str, str | None]:
        return {
            "start_date": self.start.isoformat() if self.start else None,
            "end_date": self.end.isoformat() if self.end else None,
        }


ALL_TIME = DateRange()


def parse_month(value: str) -> str:
    match = MONTH_RE.match(str(value).strip())
    if not match or not 1 <= int(match.group(2)) <= 12:
        raise InvalidInputError(f"Invalid month '{value}'. Use the format YYYY-MM, e.g. 2026-10.")
    return f"{match.group(1)}-{match.group(2)}"


def month_bounds(month: str) -> tuple[date, date]:
    month = parse_month(month)
    year, mon = int(month[:4]), int(month[5:])
    return date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])


def shift_month(month: str, delta: int) -> str:
    year, mon = int(month[:4]), int(month[5:])
    index = year * 12 + (mon - 1) + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def previous_month(month: str) -> str:
    return shift_month(month, -1)


@dataclass
class DatasetInfo:
    first_order_date: date
    last_order_date: date
    latest_full_month: str
    months: list[str] = field(default_factory=list)
    customers: int = 0
    products: int = 0
    orders: int = 0
    categories: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)

    @property
    def as_of(self) -> date:
        return self.last_order_date

    def require_month(self, month: str) -> str:
        month = parse_month(month)
        if month not in self.months:
            raise DataNotFoundError(
                f"No data for {month}. The dataset covers {self.months[0]} to {self.months[-1]}."
            )
        return month

    def relative_range(self, days: int) -> DateRange:
        return DateRange(self.as_of - timedelta(days=days - 1), self.as_of)

    def to_dict(self) -> dict:
        return {
            "first_order_date": self.first_order_date.isoformat(),
            "last_order_date": self.last_order_date.isoformat(),
            "as_of_date": self.as_of.isoformat(),
            "latest_full_month": self.latest_full_month,
            "months": self.months,
            "customers": self.customers,
            "products": self.products,
            "orders": self.orders,
            "categories": self.categories,
            "regions": self.regions,
            "channels": self.channels,
        }


def load_dataset_info(db: Database) -> DatasetInfo:
    row = db.query_one("SELECT MIN(order_date) AS first, MAX(order_date) AS last, COUNT(*) AS n FROM orders")
    if not row or not row["first"]:
        raise DataNotFoundError("The database contains no orders.")
    first, last = date.fromisoformat(row["first"]), date.fromisoformat(row["last"])
    last_day = calendar.monthrange(last.year, last.month)[1]
    current = f"{last.year:04d}-{last.month:02d}"
    latest_full = current if last.day == last_day else previous_month(current)
    months = [r["month"] for r in db.query("SELECT DISTINCT substr(order_date, 1, 7) AS month FROM orders ORDER BY month")]
    return DatasetInfo(
        first_order_date=first,
        last_order_date=last,
        latest_full_month=latest_full,
        months=months,
        customers=db.scalar("SELECT COUNT(*) FROM customers"),
        products=db.scalar("SELECT COUNT(*) FROM products"),
        orders=row["n"],
        categories=[r["category"] for r in db.query("SELECT DISTINCT category FROM products ORDER BY category")],
        regions=[r["region"] for r in db.query("SELECT DISTINCT region FROM customers ORDER BY region")],
        channels=[r["channel"] for r in db.query("SELECT DISTINCT channel FROM orders ORDER BY channel")],
    )
