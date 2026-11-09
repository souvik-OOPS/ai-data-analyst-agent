"""Response models shared by the LLM agent, the rule-based engine and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ToolCallRecord(BaseModel):
    tool: str
    input: dict
    ok: bool
    duration_ms: float
    output_preview: str | None = None
    error: str | None = None


class Grounding(BaseModel):
    """Result of checking every number in an answer against the tool outputs."""

    checked_numbers: int
    verified_numbers: int
    unverified_numbers: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grounded(self) -> bool:
        return not self.unverified_numbers


class AgentAnswer(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "unsupported", "error"]
    engine: Literal["llm", "rule_based"]
    model: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    charts: list[dict] = Field(default_factory=list)
    grounding: Grounding | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
