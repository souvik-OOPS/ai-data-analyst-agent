"""LLM rewrite of dashboard insights, verified against each insight's own evidence.

The deterministic insights (``app.analytics.insights``) are always computed
first. When an LLM is available it rewrites them for executives using
structured JSON output; any rewritten insight whose numbers do not match its
evidence is discarded and the template version is kept.
"""

from __future__ import annotations

import json
import logging

from app.agent.grounding import Evidence, check_grounding
from app.agent.llm import LLMBackend, LLMUnavailableError
from app.agent.prompts import INSIGHT_SYSTEM

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 4

INSIGHT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["id", "title", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["insights"],
        "additionalProperties": False,
    },
}


def write_insights(llm: LLMBackend | None, base: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (insights, warnings). Each insight gets ``source`` = 'llm' or 'template'."""
    insights = [{**i, "source": "template"} for i in base]
    if llm is None:
        return insights, []
    payload = [{"id": i["id"], "title": i["title"], "text": i["text"], "evidence": i["evidence"]} for i in base]
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = llm.create(
                system=INSIGHT_SYSTEM,
                messages=[{"role": "user", "content": json.dumps({"insights": payload})}],
                output_format=INSIGHT_SCHEMA,
                effort="low",
            )
            text = next(b.text for b in response.content if getattr(b, "type", None) == "text")
            rewritten = {i["id"]: i for i in json.loads(text)["insights"]}
            break
        except LLMUnavailableError as exc:
            # Same model failover as the agent: try the next model if this one is out of quota/overloaded.
            check = getattr(llm, "has_available_model", None)
            if exc.retryable and attempt + 1 < MAX_ATTEMPTS and check and check():
                continue
            return insights, [f"AI insight writing unavailable ({exc.reason}); showing computed insights."]
        except (StopIteration, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse LLM insights: %s", exc)
            return insights, ["AI insight output could not be parsed; showing computed insights."]

    rejected = 0
    for item in insights:
        candidate = rewritten.get(item["id"])
        if not candidate:
            continue
        grounding = check_grounding(candidate["text"] + " " + candidate["title"],
                                    Evidence([item["evidence"], item["text"]]))
        if grounding.grounded:
            item.update(title=candidate["title"], text=candidate["text"], source="llm")
        else:
            rejected += 1
            logger.info("Rejected LLM insight %s: unverified %s", item["id"], grounding.unverified_numbers)
    warnings = [f"{rejected} AI-written insight(s) failed number verification and were replaced by the computed "
                f"version."] if rejected else []
    return insights, warnings
