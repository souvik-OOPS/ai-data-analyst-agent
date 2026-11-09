"""SQLite access layer: schema definition, connections and query helpers.

Revenue definition used everywhere (encoded once, in the ``order_totals`` view):
    line revenue = quantity * unit_price * (1 - discount)
    only orders with status = 'completed' count toward revenue/orders/AOV.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.errors import DatabaseUnavailableError

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id   TEXT PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    email         TEXT,
    city          TEXT,
    region        TEXT NOT NULL,
    signup_date   TEXT  -- ISO date, nullable when the source had no valid value
);

CREATE TABLE products (
    product_id    TEXT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    list_price    REAL NOT NULL CHECK (list_price >= 0),
    launch_date   TEXT
);

CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT NOT NULL,  -- ISO date YYYY-MM-DD
    status        TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'returned')),
    channel       TEXT NOT NULL
);

CREATE TABLE order_items (
    order_item_id TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL REFERENCES orders(order_id),
    product_id    TEXT NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    REAL NOT NULL CHECK (unit_price >= 0),
    discount      REAL NOT NULL DEFAULT 0 CHECK (discount >= 0 AND discount < 1)
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);

-- One row per order (all statuses) with its monetary totals.
CREATE VIEW order_totals AS
SELECT o.order_id,
       o.order_date,
       substr(o.order_date, 1, 7)                          AS month,
       o.customer_id,
       c.region,
       o.channel,
       o.status,
       COUNT(oi.order_item_id)                             AS line_items,
       SUM(oi.quantity)                                    AS units,
       SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS order_revenue,
       SUM(oi.quantity * oi.unit_price * oi.discount)       AS discount_amount
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY o.order_id;

-- One row per line item of a completed order, denormalised for analysis.
CREATE VIEW sales AS
SELECT oi.order_item_id,
       o.order_id,
       o.order_date,
       substr(o.order_date, 1, 7)                    AS month,
       o.customer_id,
       c.region,
       o.channel,
       p.product_id,
       p.product_name,
       p.category,
       oi.quantity,
       oi.unit_price,
       oi.discount,
       oi.quantity * oi.unit_price * (1 - oi.discount) AS line_revenue
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed';

-- Lifetime purchasing summary per customer (completed orders only).
CREATE VIEW customer_summary AS
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       c.email,
       c.region,
       c.signup_date,
       MIN(ot.order_date)                  AS first_order_date,
       MAX(ot.order_date)                  AS last_order_date,
       COUNT(ot.order_id)                  AS orders,
       COALESCE(SUM(ot.order_revenue), 0)  AS revenue
FROM customers c
LEFT JOIN order_totals ot ON ot.customer_id = c.customer_id AND ot.status = 'completed'
GROUP BY c.customer_id;
"""

# Tables/views the read-only SQL tool may touch.
QUERYABLE_RELATIONS = frozenset(
    {"customers", "products", "orders", "order_items", "order_totals", "sales", "customer_summary"}
)

SCHEMA_DESCRIPTION = {
    "customers": "customer_id, first_name, last_name, email, city, region, signup_date",
    "products": "product_id, product_name, category, list_price, launch_date",
    "orders": "order_id, customer_id, order_date (YYYY-MM-DD), status (completed|cancelled|returned), channel",
    "order_items": "order_item_id, order_id, product_id, quantity, unit_price, discount (fraction 0-1)",
    "order_totals": "VIEW one row per order (all statuses): order_id, order_date, month (YYYY-MM), "
    "customer_id, region, channel, status, line_items, units, order_revenue, discount_amount",
    "sales": "VIEW one row per line item of COMPLETED orders: order_item_id, order_id, order_date, month, "
    "customer_id, region, channel, product_id, product_name, category, quantity, unit_price, discount, line_revenue",
    "customer_summary": "VIEW lifetime per customer (completed orders): customer_id, customer_name, email, region, "
    "signup_date, first_order_date, last_order_date, orders, revenue",
}


class Database:
    """Thin wrapper that opens a short-lived connection per unit of work.

    SQLite connections are cheap, and a connection per request avoids sharing
    one connection across FastAPI's worker threads.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    @contextmanager
    def connect(self, read_only: bool = True) -> Iterator[sqlite3.Connection]:
        if not self.exists():
            raise DatabaseUnavailableError(
                f"Database not found at {self.path}. Run `python -m app.ingestion` to build it."
            )
        uri = f"file:{self.path.as_posix()}?mode={'ro' if read_only else 'rw'}"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - environment-specific
            raise DatabaseUnavailableError(f"Could not open database: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None
