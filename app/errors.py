"""Domain exceptions. The API layer maps each one to an HTTP status code."""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-presentable errors."""

    status_code = 400
    code = "app_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"


class DataNotFoundError(AppError):
    status_code = 404
    code = "data_not_found"


class QueryRejectedError(AppError):
    """A SQL query failed validation (not read-only, unknown table, ...)."""

    status_code = 400
    code = "query_rejected"


class QueryExecutionError(AppError):
    status_code = 400
    code = "query_failed"


class DatabaseUnavailableError(AppError):
    status_code = 503
    code = "database_unavailable"


class DataIngestionError(AppError):
    status_code = 500
    code = "ingestion_failed"
