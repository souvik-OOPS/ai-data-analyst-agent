"""Data ingestion: cleaning, validation, type handling and failure modes."""

from __future__ import annotations

import sqlite3

import pytest

from app.db import Database
from app.errors import DatabaseUnavailableError, DataIngestionError
from app.ingestion import build_database
from tests.conftest import TINY_ORDERS, write_raw


def test_rows_loaded_after_cleaning(tiny_paths):
    t = tiny_paths.report.tables
    assert t["customers"].rows_read == 6 and t["customers"].rows_loaded == 4
    assert t["products"].rows_loaded == 4
    assert t["orders"].rows_read == 10 and t["orders"].rows_loaded == 6
    assert t["order_items"].rows_read == 12 and t["order_items"].rows_loaded == 8


def test_duplicates_removed(tiny_paths):
    t = tiny_paths.report.tables
    assert t["customers"].duplicates_removed == 1
    assert t["orders"].duplicates_removed == 1
    assert t["order_items"].duplicates_removed == 1


def test_invalid_rows_dropped_with_reasons(tiny_paths):
    t = tiny_paths.report.tables
    assert t["customers"].dropped == {"missing_customer_id": 1}
    assert t["orders"].dropped == {"invalid_order_date": 1, "unknown_customer_id": 1, "no_valid_line_items": 1}
    assert t["order_items"].dropped == {"order_not_loaded_or_unknown": 1, "unknown_product_id": 1,
                                        "non_positive_quantity": 1}


def test_missing_values_imputed(tiny_paths, tiny_db):
    t = tiny_paths.report.tables
    assert t["customers"].imputed["region"] == 1
    assert t["products"].imputed["category"] == 1
    assert t["orders"].imputed["channel"] == 1
    assert t["order_items"].imputed["unit_price (from product list price)"] == 1
    assert t["order_items"].imputed["discount (missing -> 0)"] == 1
    assert tiny_db.scalar("SELECT region FROM customers WHERE customer_id = 'C3'") == "Unknown"
    assert tiny_db.scalar("SELECT category FROM products WHERE product_id = 'P4'") == "Uncategorized"
    assert tiny_db.scalar("SELECT unit_price FROM order_items WHERE order_item_id = 'I4'") == 10.0
    assert tiny_db.scalar("SELECT channel FROM orders WHERE order_id = 'O5'") == "unknown"
    assert tiny_db.scalar("SELECT email FROM customers WHERE customer_id = 'C3'") is None


def test_text_and_type_normalisation(tiny_db):
    row = tiny_db.query_one("SELECT * FROM customers WHERE customer_id = 'C1'")
    assert (row["first_name"], row["last_name"], row["city"], row["email"]) == ("Alice", "Smith", "Boston",
                                                                                "alice@example.com")
    assert tiny_db.scalar("SELECT list_price FROM products WHERE product_id = 'P2'") == 1000.0
    statuses = {r["status"] for r in tiny_db.query("SELECT DISTINCT status FROM orders")}
    assert statuses == {"completed", "cancelled"}  # 'Completed ', 'COMPLETED', 'canceled' normalised
    assert tiny_db.scalar("SELECT typeof(quantity) FROM order_items LIMIT 1") == "integer"


def test_referential_integrity(tiny_db):
    assert tiny_db.scalar("SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers)") == 0
    assert tiny_db.scalar("SELECT COUNT(*) FROM order_items WHERE order_id NOT IN (SELECT order_id FROM orders)") == 0
    assert tiny_db.scalar("SELECT COUNT(*) FROM orders WHERE order_id NOT IN (SELECT order_id FROM order_items)") == 0


def test_report_stored_in_database(tiny_db):
    assert tiny_db.scalar("SELECT COUNT(*) FROM metadata WHERE key = 'ingestion_report'") == 1


def test_full_sample_has_no_duplicate_keys_or_invalid_values(full_db):
    assert full_db.scalar("SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders") == 0
    assert full_db.scalar("SELECT COUNT(*) FROM order_items WHERE quantity <= 0 OR unit_price < 0") == 0
    assert full_db.scalar("SELECT COUNT(*) FROM orders WHERE status NOT IN ('completed','cancelled','returned')") == 0


def test_missing_raw_file_raises(tmp_path):
    raw = write_raw(tmp_path / "raw")
    (raw / "orders.csv").unlink()
    with pytest.raises(DataIngestionError, match="Missing raw data file"):
        build_database(raw, tmp_path / "x.db")


def test_missing_column_raises(tmp_path):
    raw = write_raw(tmp_path / "raw", orders=TINY_ORDERS.replace("status,", "state,"))
    with pytest.raises(DataIngestionError, match="missing required columns: status"):
        build_database(raw, tmp_path / "x.db")
    assert not (tmp_path / "x.db").exists()  # no half-built database is left behind


def test_failed_rebuild_keeps_previous_database(tmp_path):
    raw = write_raw(tmp_path / "raw")
    db_path = tmp_path / "keep.db"
    build_database(raw, db_path)
    (raw / "products.csv").write_text("nonsense\n1\n", encoding="utf-8")
    with pytest.raises(DataIngestionError):
        build_database(raw, db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 6


def test_missing_database_raises_clear_error(tmp_path):
    with pytest.raises(DatabaseUnavailableError, match="python -m app.ingestion"):
        Database(tmp_path / "nope.db").query("SELECT 1")
