"""Request bodies for the API (query parameters are validated in the route signatures)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., max_length=1000)
    answer: str = Field(..., max_length=8000)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=1000, description="Natural-language business question.")
    history: list[HistoryTurn] = Field(default_factory=list, max_length=10,
                                       description="Previous turns (oldest first) for follow-up questions.")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SqlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, max_length=2000)
