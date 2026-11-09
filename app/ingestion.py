"""Data ingestion: load raw CSV extracts, clean and validate them, and build the SQLite database.

Every cleaning decision is counted in an :class:`IngestionReport` so data-quality
problems are visible (``GET /api/dataset``) instead of silently disappearing.

Run directly to (re)build the database:
    python -m app.ingestion
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.db import SCHEMA
from app.errors import DataIngestionError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "customers": ["customer_id", "first_name", "last_name", "email", "city", "region", "signup_date"],
    "products": ["product_id", "product_name", "category", "list_price", "launch_date"],
    "orders": ["order_id", "customer_id", "order_date", "status", "channel"],
    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount"],
}
VALID_STATUSES = {"completed", "cancelled", "returned"}
STATUS_SYNONYMS = {"canceled": "cancelled", "complete": "completed", "return": "returned"}
DATE_FORMAT = "%Y-%m-%d"


@dataclass
class TableReport:
    rows_read: int = 0
    rows_loaded: int = 0
    duplicates_removed: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    imputed: dict[str, int] = field(default_factory=dict)
    normalized: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def impute(self, column: str, n: int) -> None:
        if n:
            self.imputed[column] = self.imputed.get(column, 0) + int(n)

    def normalize(self, column: str, n: int) -> None:
        if n:
            self.normalized[column] = self.normalized.get(column, 0) + int(n)


@dataclass
class IngestionReport:
    tables: dict[str, TableReport] = field(default_factory=lambda: {t: TableReport() for t in REQUIRED_COLUMNS})
    built_at: str = ""
    first_order_date: str | None = None
    last_order_date: str | None = None

    def to_dict(self) -> dict:
        return {
            "built_at": self.built_at,
            "first_order_date": self.first_order_date,
            "last_order_date": self.last_order_date,
            "tables": {name: asdict(rep) for name, rep in self.tables.items()},
        }


# --------------------------------------------------------------------------- helpers


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim, collapse internal whitespace and turn empty strings into missing values."""
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.mask(cleaned == "")


def _count_changed(before: pd.Series, after: pd.Series) -> int:
    both = before.notna() & after.notna()
    return int((before[both] != after[both]).sum())


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")


def _prepare(df: pd.DataFrame, table: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS[table] if c not in df.columns]
    if missing:
        raise DataIngestionError(f"{table}.csv is missing required columns: {', '.join(missing)}")
    df = df[REQUIRED_COLUMNS[table]]
    for col in df.columns:
        df[col] = _clean_text(df[col])
    return df


def _dedupe(df: pd.DataFrame, key: str, rep: TableReport) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[key], keep="first")
    rep.duplicates_removed += before - len(df)
    return df


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str, rep: TableReport) -> pd.DataFrame:
    rep.drop(reason, int(mask.sum()))
    return df[~mask]


# --------------------------------------------------------------------------- cleaners


def clean_customers(raw: pd.DataFrame, rep: TableReport) -> pd.DataFrame:
    rep.rows_read = len(raw)
    df = _prepare(raw, "customers")
    df["customer_id"] = df["customer_id"].str.upper()
    df = _drop(df, df["customer_id"].isna(), "missing_customer_id", rep)

    for col in ("first_name", "last_name", "city"):
        titled = df[col].str.title()
        rep.normalize(col, _count_changed(df[col], titled))
        df[col] = titled
    lowered = df["email"].str.lower()
    rep.normalize("email", _count_changed(df["email"], lowered))
    df["email"] = lowered

    rep.impute("first_name", int(df["first_name"].isna().sum()))
    rep.impute("last_name", int(df["last_name"].isna().sum()))
    df["first_name"] = df["first_name"].fillna("Unknown")
    df["last_name"] = df["last_name"].fillna("Unknown")
    rep.impute("region", int(df["region"].isna().sum()))
    df["region"] = df["region"].fillna("Unknown").str.title()

    signup = _parse_dates(df["signup_date"])
    rep.impute("signup_date (invalid -> null)", int((signup.isna() & df["signup_date"].notna()).sum()))
    df["signup_date"] = signup.dt.strftime(DATE_FORMAT)

    df = _dedupe(df, "customer_id", rep)
    # Missing emails are kept (the customer is still valid) but reported.
    rep.impute("email (missing, kept as null)", int(df["email"].isna().sum()))
    return df


def clean_products(raw: pd.DataFrame, rep: TableReport) -> pd.DataFrame:
    rep.rows_read = len(raw)
    df = _prepare(raw, "products")
    df["product_id"] = df["product_id"].str.upper()
    df = _drop(df, df["product_id"].isna(), "missing_product_id", rep)
    df = _drop(df, df["product_name"].isna(), "missing_product_name", rep)

    price_text = df["list_price"].str.replace(r"[$,\s]", "", regex=True)
    rep.normalize("list_price (currency text -> number)", int(df["list_price"].str.contains(r"[$,]", regex=True).fillna(False).sum()))
    price = pd.to_numeric(price_text, errors="coerce")
    df = _drop(df, price.isna() | (price < 0), "invalid_list_price", rep)
    df["list_price"] = price[df.index].astype(float)

    rep.impute("category", int(df["category"].isna().sum()))
    df["category"] = df["category"].fillna("Uncategorized")
    df["launch_date"] = _parse_dates(df["launch_date"]).dt.strftime(DATE_FORMAT)
    return _dedupe(df, "product_id", rep)


def clean_orders(raw: pd.DataFrame, customer_ids: set[str], rep: TableReport) -> pd.DataFrame:
    rep.rows_read = len(raw)
    df = _prepare(raw, "orders")
    df["order_id"] = df["order_id"].str.upper()
    df["customer_id"] = df["customer_id"].str.upper()
    df = _drop(df, df["order_id"].isna(), "missing_order_id", rep)

    status = df["status"].str.lower()
    status = status.replace(STATUS_SYNONYMS)
    rep.normalize("status", _count_changed(df["status"], status))
    df["status"] = status
    df = _drop(df, ~df["status"].isin(VALID_STATUSES), "invalid_status", rep)

    dates = _parse_dates(df["order_date"])
    df = _drop(df, dates.isna(), "invalid_order_date", rep)
    df["order_date"] = dates[df.index].dt.strftime(DATE_FORMAT)

    df = _drop(df, df["customer_id"].isna(), "missing_customer_id", rep)
    df = _drop(df, ~df["customer_id"].isin(customer_ids), "unknown_customer_id", rep)

    rep.impute("channel", int(df["channel"].isna().sum()))
    df["channel"] = df["channel"].fillna("unknown").str.lower()
    return _dedupe(df, "order_id", rep)


def clean_order_items(
    raw: pd.DataFrame, order_ids: set[str], products: pd.DataFrame, rep: TableReport
) -> pd.DataFrame:
    rep.rows_read = len(raw)
    df = _prepare(raw, "order_items")
    for col in ("order_item_id", "order_id", "product_id"):
        df[col] = df[col].str.upper()
    df = _drop(df, df["order_item_id"].isna(), "missing_order_item_id", rep)
    df = _dedupe(df, "order_item_id", rep)

    df = _drop(df, ~df["order_id"].isin(order_ids), "order_not_loaded_or_unknown", rep)
    df = _drop(df, ~df["product_id"].isin(set(products["product_id"])), "unknown_product_id", rep)

    qty = pd.to_numeric(df["quantity"], errors="coerce")
    df = _drop(df, qty.isna(), "missing_quantity", rep)
    qty = qty[df.index]
    df = _drop(df, qty <= 0, "non_positive_quantity", rep)
    qty = qty[df.index]
    df = _drop(df, qty != qty.round(), "fractional_quantity", rep)
    df["quantity"] = qty[df.index].astype(int)

    price = pd.to_numeric(df["unit_price"], errors="coerce")
    list_price = df["product_id"].map(products.set_index("product_id")["list_price"])
    rep.impute("unit_price (from product list price)", int(price.isna().sum()))
    price = price.fillna(list_price)
    df = _drop(df, price < 0, "negative_unit_price", rep)
    df["unit_price"] = price[df.index].astype(float)

    discount = pd.to_numeric(df["discount"], errors="coerce")
    rep.impute("discount (missing -> 0)", int(discount.isna().sum()))
    out_of_range = discount.notna() & ((discount < 0) | (discount >= 1))
    rep.impute("discount (out of range -> 0)", int(out_of_range.sum()))
    df["discount"] = discount.mask(out_of_range).fillna(0.0).astype(float)
    return df


# --------------------------------------------------------------------------- pipeline


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for table in REQUIRED_COLUMNS:
        path = Path(raw_dir) / f"{table}.csv"
        if not path.is_file():
            raise DataIngestionError(f"Missing raw data file: {path}")
        try:
            frames[table] = pd.read_csv(path, dtype=str, keep_default_na=True)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DataIngestionError(f"Could not parse {path.name}: {exc}") from exc
    return frames


def clean_all(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], IngestionReport]:
    report = IngestionReport()
    t = report.tables
    customers = clean_customers(frames["customers"], t["customers"])
    products = clean_products(frames["products"], t["products"])
    orders = clean_orders(frames["orders"], set(customers["customer_id"]), t["orders"])
    items = clean_order_items(frames["order_items"], set(orders["order_id"]), products, t["order_items"])

    # An order whose every line was invalid has no revenue basis -> drop it.
    orders = _drop(orders, ~orders["order_id"].isin(set(items["order_id"])), "no_valid_line_items", t["orders"])

    cleaned = {"customers": customers, "products": products, "orders": orders, "order_items": items}
    for name, df in cleaned.items():
        t[name].rows_loaded = len(df)
    if not orders.empty:
        report.first_order_date = str(orders["order_date"].min())
        report.last_order_date = str(orders["order_date"].max())
    return cleaned, report


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    subset = df[columns].astype(object)
    subset = subset.where(subset.notna(), None)
    return list(subset.itertuples(index=False, name=None))


def write_database(cleaned: dict[str, pd.DataFrame], report: IngestionReport, db_path: Path) -> None:
    """Write to a temporary file, then atomically swap it in, so readers never see a half-built DB."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".building")
    tmp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(SCHEMA)
        for table, columns in REQUIRED_COLUMNS.items():
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                _rows(cleaned[table], columns),
            )
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DataIngestionError(f"Foreign key violations after cleaning: {violations[:5]}")
        conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("ingestion_report", json.dumps(report.to_dict())))
        conn.commit()
    except sqlite3.Error as exc:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise DataIngestionError(f"Failed to write database: {exc}") from exc
    except Exception:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp_path, db_path)


def build_database(raw_dir: Path, db_path: Path) -> IngestionReport:
    frames = read_raw(raw_dir)
    cleaned, report = clean_all(frames)
    report.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_database(cleaned, report, db_path)
    logger.info("Database built at %s", db_path)
    return report


def main() -> None:  # pragma: no cover - CLI wrapper
    from app.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    report = build_database(settings.raw_data_dir, settings.database_path)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
