"""Shared fixtures.

Two datasets are used:
- ``tiny``: a hand-written dataset (with deliberate data-quality problems) whose
  correct answers were worked out by hand - see the comments in TINY_* below.
- ``full``: the real sample data in data/raw, built into a temporary database and
  cross-checked against independent pandas computations.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.agent.tools import ToolContext, ToolRegistry
from app.analytics.periods import load_dataset_info
from app.config import PROJECT_ROOT, Settings
from app.db import Database
from app.ingestion import build_database
from app.main import create_app

# --------------------------------------------------------------------------- tiny dataset
# Completed orders after cleaning: O1 (C1, 2025-01-10, $30), O2 (C1, 2025-02-15, $900),
# O3 (C2, 2025-02-20, $30), O5 (C3, 2025-03-05, $50)  ->  revenue $1,010, 4 orders, AOV $252.50.
# Cancelled orders: O4, O8. Latest order date 2025-03-20, so the latest *complete* month is 2025-02.

TINY_CUSTOMERS = """\
customer_id,first_name,last_name,email,city,region,signup_date
C1,alice,smith,ALICE@EXAMPLE.COM,  boston ,North,2025-01-01
C2,Bob,Jones,bob@example.com,Miami,South,2025-01-05
C3,Cara,Lee,,Seattle,,2025-02-01
C4,Dan,Wu,dan@example.com,Denver,Central,2025-02-10
C2,Bob,Jones,bob@example.com,Miami,South,2025-01-05
,No,Id,x@example.com,Nowhere,North,2025-01-01
"""

TINY_PRODUCTS = """\
product_id,product_name,category,list_price,launch_date
P1,Widget,Gadgets,10.00,2025-01-01
P2,Gizmo,Gadgets,"$1,000.00",2025-01-01
P3,Book A,Books,20,2025-01-01
P4,Mystery Item,,5,2025-01-01
"""

TINY_ORDERS = """\
order_id,customer_id,order_date,status,channel
O1,C1,2025-01-10,completed,web
O2,C1,2025-02-15,Completed ,mobile
O3,C2,2025-02-20,COMPLETED,web
O4,C2,2025-02-21,cancelled,web
O5,C3,2025-03-05,completed,
O6,C9,2025-03-06,completed,web
O7,C3,not-a-date,completed,web
O3,C2,2025-02-20,COMPLETED,web
O8,C1,2025-03-20,canceled,web
O9,C2,2025-03-25,completed,web
"""

TINY_ITEMS = """\
order_item_id,order_id,product_id,quantity,unit_price,discount
I1,O1,P1,2,10,0
I2,O1,P3,1,20,0.5
I3,O2,P2,1,1000,0.1
I4,O3,P1,3,,0
I5,O4,P2,1,1000,0
I6,O5,P3,2,20,
I7,O5,P1,1,10,0
I8,O6,P1,1,10,0
I9,O9,P1,-1,10,0
I10,O9,P9,1,10,0
I1,O1,P1,2,10,0
I11,O8,P1,1,10,0
"""


def write_raw(directory: Path, customers=TINY_CUSTOMERS, products=TINY_PRODUCTS, orders=TINY_ORDERS,
              items=TINY_ITEMS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (("customers", customers), ("products", products), ("orders", orders),
                       ("order_items", items)):
        (directory / f"{name}.csv").write_text(textwrap.dedent(text), encoding="utf-8")
    return directory


@pytest.fixture(scope="session")
def tiny_paths(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("tiny")
    raw = write_raw(root / "raw")
    db_path = root / "tiny.db"
    report = build_database(raw, db_path)
    return SimpleNamespace(raw=raw, db_path=db_path, report=report)


@pytest.fixture(scope="session")
def tiny_db(tiny_paths) -> Database:
    return Database(tiny_paths.db_path)


@pytest.fixture(scope="session")
def tiny_info(tiny_db):
    return load_dataset_info(tiny_db)


# --------------------------------------------------------------------------- full sample dataset


@pytest.fixture(scope="session")
def full_db_path(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("full") / "ecommerce.db"
    build_database(PROJECT_ROOT / "data" / "raw", db_path)
    return db_path


@pytest.fixture(scope="session")
def full_db(full_db_path) -> Database:
    return Database(full_db_path)


@pytest.fixture(scope="session")
def full_info(full_db):
    return load_dataset_info(full_db)


@pytest.fixture(scope="session")
def registry(full_db, full_info) -> ToolRegistry:
    return ToolRegistry(ToolContext(full_db, full_info))


@pytest.fixture(scope="session")
def ref(full_db_path) -> SimpleNamespace:
    """Independent pandas reference built from the base tables (not the SQL views)."""
    with sqlite3.connect(full_db_path) as conn:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        products = pd.read_sql("SELECT * FROM products", conn)
        orders = pd.read_sql("SELECT * FROM orders", conn)
        items = pd.read_sql("SELECT * FROM order_items", conn)
    items["revenue"] = items["quantity"] * items["unit_price"] * (1 - items["discount"])
    completed = orders[orders["status"] == "completed"]
    lines = items.merge(completed, on="order_id").merge(products, on="product_id")
    lines["month"] = lines["order_date"].str[:7]
    order_rev = lines.groupby(["order_id", "customer_id", "order_date", "month"], as_index=False)["revenue"].sum()
    return SimpleNamespace(customers=customers, products=products, orders=orders, items=items,
                           lines=lines, order_rev=order_rev)


# --------------------------------------------------------------------------- API client


def make_settings(db_path: Path, raw_dir: Path) -> Settings:
    return Settings(_env_file=None, database_path=db_path, raw_data_dir=raw_dir, llm_enabled=False,
                    anthropic_api_key=None)


@pytest.fixture(scope="session")
def client(full_db_path) -> TestClient:
    app = create_app(make_settings(full_db_path, PROJECT_ROOT / "data" / "raw"))
    with TestClient(app) as c:
        yield c
