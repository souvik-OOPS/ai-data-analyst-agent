"""The read-only SQL tool must never modify data or read outside the whitelisted relations."""

from __future__ import annotations

import pytest

from app.errors import QueryExecutionError, QueryRejectedError
from app.sql_guard import run_read_only_query


@pytest.mark.parametrize("sql", [
    "SELECT category, SUM(line_revenue) AS revenue FROM sales GROUP BY category",
    "with x as (select region, count(*) n from customers group by region) select * from x;",
    "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 5) SELECT * FROM c",
    "WITH m AS (SELECT month, SUM(order_revenue) r FROM order_totals GROUP BY month) SELECT COUNT(*) FROM m",
    "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id) AS t",
])
def test_allowed_queries(full_db, sql):
    result = run_read_only_query(full_db, sql)
    assert result["row_count"] > 0 and result["columns"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "DELETE FROM orders",
    "UPDATE orders SET status = 'completed'",
    "INSERT INTO customers SELECT * FROM customers",
    "SELECT 1; DELETE FROM orders",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'other.db' AS other",
    "SELECT * FROM customers -- comment",
    "SELECT * FROM customers /* comment */",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM metadata",
    "SELECT * FROM pragma_table_info('customers')",
    "SELECT load_extension('evil')",
    "",
    "   ",
])
def test_rejected_queries(full_db, sql):
    with pytest.raises(QueryRejectedError):
        run_read_only_query(full_db, sql)


def test_data_unchanged_after_attacks(full_db):
    before = full_db.scalar("SELECT COUNT(*) FROM orders")
    for sql in ("DELETE FROM orders", "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x"):
        with pytest.raises((QueryRejectedError, QueryExecutionError)):
            run_read_only_query(full_db, sql)
    assert full_db.scalar("SELECT COUNT(*) FROM orders") == before


def test_row_cap(full_db):
    result = run_read_only_query(full_db, "SELECT order_id FROM orders", max_rows=50)
    assert result["row_count"] == 50 and result["truncated"] is True


def test_sql_error_is_reported(full_db):
    with pytest.raises(QueryExecutionError, match="no such column"):
        run_read_only_query(full_db, "SELECT nonexistent FROM customers")


def test_runaway_query_is_stopped(full_db):
    with pytest.raises(QueryExecutionError, match="time budget"):
        run_read_only_query(full_db, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) "
                                     "SELECT COUNT(*) FROM c")
