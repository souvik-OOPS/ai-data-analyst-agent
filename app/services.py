"""Application wiring: builds the database, dataset info, tool registry, LLM client and agent once."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from app.agent.agent import AnalystAgent
from app.agent.insight_writer import write_insights
from app.agent.llm import ClaudeClient, LLMBackend
from app.agent.tools import ToolContext, ToolRegistry
from app.analytics import compute_insights, load_dataset_info
from app.analytics.periods import DatasetInfo
from app.config import Settings
from app.db import Database
from app.ingestion import build_database

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    db: Database
    info: DatasetInfo
    registry: ToolRegistry
    agent: AnalystAgent
    llm: LLMBackend | None
    _insights_cache: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def insights(self, use_llm: bool) -> dict:
        """Insights are cached: the dataset is static, and LLM rewriting is the slow part.
        A result is only cached when the LLM call succeeded, so a temporary outage is retried later."""
        key = "llm" if (use_llm and self.llm) else "template"
        with self._lock:
            if key in self._insights_cache:
                return self._insights_cache[key]
            base = compute_insights(self.db, self.info)
            items, warnings = write_insights(self.llm if key == "llm" else None, base)
            llm_written = any(i["source"] == "llm" for i in items)
            result = {"insights": items, "warnings": warnings, "engine": "llm" if llm_written else "template",
                      "as_of_date": self.info.as_of.isoformat()}
            if key == "template" or llm_written:
                self._insights_cache[key] = result
            return result


def create_llm(settings: Settings) -> LLMBackend | None:
    """Pick the LLM backend from settings: Gemini, Claude, or None (offline rule-based mode)."""
    provider = settings.resolved_provider
    if provider == "gemini":
        from app.agent.gemini import GeminiClient  # imported lazily: only needed when used

        return GeminiClient(settings)
    if provider == "anthropic":
        return ClaudeClient(settings)
    return None


def build_services(settings: Settings, llm: LLMBackend | None = None) -> Services:
    db = Database(settings.database_path)
    if settings.rebuild_db_on_start or not db.exists():
        logger.info("Building database from %s", settings.raw_data_dir)
        build_database(settings.raw_data_dir, settings.database_path)
    info = load_dataset_info(db)
    registry = ToolRegistry(ToolContext(db, info))
    if llm is None:
        llm = create_llm(settings)
    agent = AnalystAgent(registry, info, llm, settings.llm_max_tool_iterations, settings.max_question_chars)
    logger.info("Agent mode: %s%s", agent.mode, f" ({llm.provider}: {llm.model})" if llm else "")
    return Services(settings=settings, db=db, info=info, registry=registry, agent=agent, llm=llm)
