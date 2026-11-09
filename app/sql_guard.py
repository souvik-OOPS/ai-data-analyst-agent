"""Guarded, read-only execution of ad-hoc SQL (used by the agent's ``run_sql`` tool).

Defence in depth - each layer alone would block writes:
1. static checks: single statement, must start with SELECT/WITH, no comments, length cap
2. the connection is opened read-only (``mode=ro``)
3. an SQLite authorizer allows only reads of whitelisted tables/views and safe functions
4. a progress handler aborts long-running queries; results are capped at ``max_rows``
"""

from __future__ import annotations

import re
import sqlite3
import time

from app.db import QUERYABLE_RELATIONS, Database
from app.errors import QueryExecutionError, QueryRejectedError

MAX_SQL_CHARS = 2000
DEFAULT_MAX_ROWS = 200
TIME_BUDGET_SECONDS = 3.0
_FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
_STARTS_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def validate_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise QueryRejectedError("Query is empty.")
    sql = sql.strip()
    if len(sql) > MAX_SQL_CHARS:
        raise QueryRejectedError(f"Query is too long (max {MAX_SQL_CHARS} characters).")
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    if ";" in sql:
        raise QueryRejectedError("Only a single SQL statement is allowed.")
    if "--" in sql or "/*" in sql:
        raise QueryRejectedError("SQL comments are not allowed.")
    if not _STARTS_READ_ONLY.match(sql):
        raise QueryRejectedError("Only read-only SELECT (or WITH ... SELECT) queries are allowed.")
    if not sqlite3.complete_statement(sql + ";"):
        raise QueryRejectedError("Query is not a complete SQL statement.")
    return sql


def _make_authorizer(schema_objects: set[str]):
    """Allow reads of whitelisted relations and of CTE/subquery names; deny everything else.

    SQLite also reports reads of a CTE under the CTE's own name (e.g. ``COUNT(*) FROM c``),
    so names that are not real schema objects are allowed. Real tables/views outside the
    whitelist, SQLite internals and PRAGMA table-valued functions are denied.
    """

    def authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            name = (arg1 or "").lower()
            if name in QUERYABLE_RELATIONS:
                return sqlite3.SQLITE_OK
            if name.startswith(("sqlite_", "pragma_")) or name in schema_objects:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in _FORBIDDEN_FUNCTIONS else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorizer


def run_read_only_query(db: Database, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    sql = validate_sql(sql)
    started = time.monotonic()
    timed_out = False

    def over_budget() -> int:
        nonlocal timed_out
        timed_out = time.monotonic() - started > TIME_BUDGET_SECONDS
        return int(timed_out)

    with db.connect(read_only=True) as conn:
        schema_objects = {r[0].lower() for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        conn.set_authorizer(_make_authorizer(schema_objects))
        conn.set_progress_handler(over_budget, 10_000)
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            # Detect the timeout from our own flag rather than by parsing SQLite's message.
            if timed_out:
                raise QueryExecutionError(f"Query exceeded the {TIME_BUDGET_SECONDS:.0f}s time budget.") from exc
            message = str(exc)
            if "not authorized" in message or "prohibited" in message:
                raise QueryRejectedError(
                    "Query touches a table, view or operation that is not allowed. "
                    f"Allowed relations: {', '.join(sorted(QUERYABLE_RELATIONS))}."
                ) from exc
            raise QueryExecutionError(f"SQL error: {message}") from exc
        columns = [d[0] for d in cursor.description or []]
    truncated = len(rows) > max_rows
    return {
        "columns": columns,
        "rows": [dict(zip(columns, tuple(r), strict=True)) for r in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": truncated,
        "sql": sql,
    }
