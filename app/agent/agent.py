"""The analyst agent: Claude tool-use loop with grounding verification and fallbacks.

Flow for one question
---------------------
1. If no LLM is configured, answer with the rule-based engine (``engine="rule_based"``).
2. Otherwise run a manual tool-use loop with Claude: the model picks tools, the
   registry validates and executes them, results go back to the model, until it
   writes a final answer (bounded by ``llm_max_tool_iterations``).
3. Verify the final answer: every number must match a tool result
   (:mod:`app.agent.grounding`). On failure, ask the model once to correct itself.
   If it still fails, fall back to the rule-based answer when one exists, or
   show the LLM answer with an explicit warning listing the unverified figures.
4. Any LLM/API failure (auth, rate limit, timeout, refusal, ...) falls back to
   the rule-based engine with a warning, so the user still gets an answer.
"""

from __future__ import annotations

import logging
import time

from app.agent.grounding import Evidence, check_grounding
from app.agent.llm import LLMBackend, LLMUnavailableError
from app.agent.models import AgentAnswer, Grounding, ToolCallRecord
from app.agent.prompts import CORRECTION_NO_TOOLS, CORRECTION_UNGROUNDED, build_system_prompt
from app.agent.router import EXAMPLE_QUESTIONS, RuleBasedAnalyst
from app.agent.tools import UNSUPPORTED_TOOL, ToolRegistry, ToolResult
from app.analytics.periods import DatasetInfo
from app.errors import InvalidInputError

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4
MAX_FAILOVERS = 3


def _can_failover(llm) -> bool:
    """True if the backend supports model failover and still has a model available."""
    check = getattr(llm, "has_available_model", None)
    return bool(check and check())


class AnalystAgent:
    def __init__(self, registry: ToolRegistry, info: DatasetInfo, llm: LLMBackend | None = None,
                 max_iterations: int = 8, max_question_chars: int = 1000):
        self.registry = registry
        self.info = info
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_question_chars = max_question_chars
        self.rules = RuleBasedAnalyst(registry, info)
        self.system_prompt = build_system_prompt(info)
        self.tool_schemas = registry.schemas()

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "rule_based"

    def ask(self, question: str, history: list[dict] | None = None) -> AgentAnswer:
        question = (question or "").strip()
        if not question:
            raise InvalidInputError("Question must not be empty.")
        if len(question) > self.max_question_chars:
            raise InvalidInputError(f"Question is too long (max {self.max_question_chars} characters).")

        if self.llm is None:
            answer = self.rules.answer(question)
            answer.warnings.append("Offline mode: no LLM API key is configured, so the rule-based engine answered. "
                                   "It understands a narrower range of phrasings.")
            return answer

        started = time.perf_counter()
        skipped: list[str] = []
        while True:
            try:
                answer = self._ask_llm(question, history or [])
                break
            except LLMUnavailableError as exc:
                logger.warning("LLM unavailable (%s): %s", exc.kind, exc.reason)
                # Model failover: if another model may succeed (quota/overload), restart the question on it.
                if exc.retryable and len(skipped) < MAX_FAILOVERS and _can_failover(self.llm):
                    skipped.append(exc.reason)
                    continue
                answer = self.rules.answer(question)
                answer.warnings.insert(0, f"The AI model was unavailable ({exc.reason}), so the rule-based engine "
                                          f"answered instead.")
                break
        if skipped and answer.engine == "llm":
            answer.warnings.append(f"Answered by fallback model {answer.model} ({'; '.join(skipped)}).")
        answer.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return answer

    # ------------------------------------------------------------------ LLM loop

    def _ask_llm(self, question: str, history: list[dict]) -> AgentAnswer:
        messages: list[dict] = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": question})

        results: list[ToolResult] = []
        warnings: list[str] = []
        model_used = self.llm.model
        evidence_sources: list = [self.info.to_dict()]
        corrected_grounding = corrected_no_tools = False
        text, grounding = "", None

        for _ in range(self.max_iterations):
            response = self.llm.create(system=self.system_prompt, messages=messages, tools=self.tool_schemas)
            model_used = getattr(response, "model", None) or model_used  # the model that actually answered
            # Append the assistant turn unchanged (thinking blocks included): history stays append-only.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if tool_uses:
                tool_results, unsupported = [], None
                for block in tool_uses:
                    result = self.registry.execute(block.name, block.input)
                    if block.name == UNSUPPORTED_TOOL and result.ok:
                        unsupported = block.input
                    else:
                        results.append(result)
                        if result.ok:
                            evidence_sources.append(result.output)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": result.content_for_llm(), "is_error": not result.ok})
                if unsupported is not None:
                    return self._unsupported_answer(question, unsupported, results, model_used)
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "pause_turn":
                continue
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
            if response.stop_reason == "max_tokens":
                warnings.append("The answer hit the output length limit and may be incomplete.")

            if not results and not corrected_no_tools and self.rules.detect(question).name not in (
                    "unsupported", "invalid", "unknown"):
                corrected_no_tools = True
                messages.append({"role": "user", "content": CORRECTION_NO_TOOLS})
                continue

            grounding = check_grounding(text, Evidence(evidence_sources), question + " " + _history_text(history))
            if not grounding.grounded and not corrected_grounding:
                corrected_grounding = True
                logger.info("Unverified figures %s; asking the model to correct", grounding.unverified_numbers)
                messages.append({"role": "user", "content": CORRECTION_UNGROUNDED.format(
                    numbers=", ".join(grounding.unverified_numbers))})
                continue
            break
        else:
            return self._fallback(question, results, model_used,
                                  "The analysis did not finish within the allowed number of steps.")

        if grounding is not None and not grounding.grounded:
            return self._fallback(question, results, model_used, "The AI answer contained figures that could not be verified "
                                  f"against the data ({', '.join(grounding.unverified_numbers)}).",
                                  llm_text=text, grounding=grounding)

        return AgentAnswer(question=question, answer=text or "I could not produce an answer.",
                           status="answered" if text else "error", engine="llm", model=model_used,
                           tool_calls=_records(results), charts=[r.chart for r in results if r.chart],
                           grounding=grounding, warnings=warnings)

    def _fallback(self, question: str, results: list[ToolResult], model: str, reason: str, llm_text: str = "",
                  grounding: Grounding | None = None) -> AgentAnswer:
        rule_answer = self.rules.answer(question)
        if rule_answer.status == "answered":
            rule_answer.warnings.insert(0, f"{reason} Showing the verified rule-based answer instead.")
            rule_answer.tool_calls = _records(results) + rule_answer.tool_calls
            return rule_answer
        return AgentAnswer(
            question=question,
            answer=llm_text or "I could not complete this analysis. Please try rephrasing the question.",
            status="answered" if llm_text else "error", engine="llm", model=model,
            tool_calls=_records(results), charts=[r.chart for r in results if r.chart], grounding=grounding,
            warnings=[f"{reason} Treat any unverified figures with caution."],
        )

    def _unsupported_answer(self, question: str, args: dict, results: list[ToolResult], model: str) -> AgentAnswer:
        suggestion = args.get("suggestion")
        examples = "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS[:4])
        text = f"**I can't answer that from this dataset.** {args.get('reason', '').strip()}"
        text += f"\n\nYou could ask instead: *{suggestion.strip()}*" if suggestion else f"\n\nFor example:\n{examples}"
        return AgentAnswer(question=question, answer=text, status="unsupported", engine="llm", model=model,
                           tool_calls=_records(results))


def _records(results: list[ToolResult]) -> list[ToolCallRecord]:
    return [ToolCallRecord(tool=r.name, input=r.input, ok=r.ok, duration_ms=r.duration_ms,
                           output_preview=r.preview() if r.ok else None, error=r.error) for r in results]


def _history_text(history: list[dict]) -> str:
    return " ".join(f"{t.get('question', '')} {t.get('answer', '')}" for t in history[-MAX_HISTORY_TURNS:])
