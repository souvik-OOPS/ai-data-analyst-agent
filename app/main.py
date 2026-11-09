"""FastAPI application factory.

Run with:  uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000  (dashboard)   http://127.0.0.1:8000/docs  (API docs)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.agent.llm import LLMBackend
from app.api.routes import router
from app.config import Settings, get_settings
from app.errors import AppError
from app.services import build_services

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("app")


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def create_app(settings: Settings | None = None, llm: LLMBackend | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings, llm)
        yield

    app = FastAPI(
        title="AI Data Analyst & BI Agent",
        version=__version__,
        description="Ask business questions in natural language; answers are computed from the order database.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
                   for e in exc.errors()]
        summary = "; ".join(f"{d['field']}: {d['message']}" for d in details)
        return _error(422, "invalid_input", f"Invalid request - {summary}", details)

    @app.exception_handler(sqlite3.Error)
    async def _db_error(_: Request, exc: sqlite3.Error):
        logger.exception("Database error")
        return _error(503, "database_error", "The database could not process the request.")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return _error(500, "internal_error", "An unexpected error occurred.")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
