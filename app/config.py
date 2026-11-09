"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Data ---------------------------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    database_path: Path = PROJECT_ROOT / "data" / "ecommerce.db"
    rebuild_db_on_start: bool = False

    # --- LLM provider ------------------------------------------------------------
    # auto = Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else offline.
    llm_provider: Literal["auto", "gemini", "anthropic", "none"] = "auto"
    llm_enabled: bool = True  # set false to force the offline rule-based engine
    llm_max_tokens: int = 16000
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2
    llm_max_tool_iterations: int = Field(8, ge=1, le=20)

    # Google Gemini
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    # Tried in order when the current model is out of quota or overloaded (each model has its own quota).
    gemini_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.8-flash,gemini-flash-latest"

    # Anthropic Claude
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Server-side refusal fallback (beta): if a request is declined by a safety
    # classifier, the API re-runs it on a fallback model inside the same call.
    llm_refusal_fallback: bool = True

    # --- API -------------------------------------------------------------------
    max_question_chars: int = 1000

    @property
    def gemini_models(self) -> list[str]:
        """Primary Gemini model followed by the fallbacks, without duplicates."""
        fallbacks = [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]
        return list(dict.fromkeys([self.gemini_model, *fallbacks]))

    @property
    def resolved_provider(self) -> str:
        """Which LLM provider will actually be used: 'gemini', 'anthropic' or 'none' (offline)."""
        if not self.llm_enabled or self.llm_provider == "none":
            return "none"
        if self.llm_provider == "gemini":
            return "gemini" if self.gemini_api_key else "none"
        if self.llm_provider == "anthropic":
            return "anthropic" if self.anthropic_api_key else "none"
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "anthropic"
        return "none"

    @property
    def llm_configured(self) -> bool:
        return self.resolved_provider != "none"


@lru_cache
def get_settings() -> Settings:
    return Settings()
