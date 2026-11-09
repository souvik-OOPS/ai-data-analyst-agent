"""Google Gemini backend for the analyst agent.

Model/provider: Google Gemini (default ``gemini-3.5-flash``, configurable via GEMINI_MODEL),
called through Google's official ``google-genai`` SDK (Gemini Developer API).

It implements the same ``create()`` contract as :class:`app.agent.llm.ClaudeClient`, so the
agent loop does not care which provider it talks to:

* input:  the conversation as provider-neutral messages - text, ``tool_use`` and
  ``tool_result`` blocks - which are converted to Gemini ``contents`` on every call;
* output: an object with ``.content`` (text / tool_use blocks) and ``.stop_reason``.

Gemini 3 models attach "thought signatures" to their function calls, and these must be
sent back unchanged. The model's original reply is therefore kept in a ``gemini_raw``
block (which the agent ignores) and is replayed as-is when the conversation continues.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app.agent.llm import LLMUnavailableError
from app.config import Settings

logger = logging.getLogger(__name__)

LOCAL_ID_PREFIX = "local-"  # ids we invent when the API does not return one
RETRY_STATUS_CODES = [408, 500, 502, 504]
BLOCKED_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE"}


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class RawGeminiContent:
    """The model's turn exactly as Gemini returned it (including thought signatures),
    plus which model produced it - a conversation must stay on that model."""

    content: Any
    model: str = ""
    type: str = "gemini_raw"


@dataclass
class GeminiResponse:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Any = None
    model: str = ""  # the model that actually produced this response


# --------------------------------------------------------------------------- conversion


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _parse_tool_output(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    """Convert provider-neutral messages into Gemini ``contents``."""
    contents: list[types.Content] = []
    tool_names: dict[str, str] = {}  # tool_use_id -> tool name (function responses need the name)
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        content = message["content"]
        if isinstance(content, str):
            contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
            continue
        for block in content:
            if _block_get(block, "type") == "tool_use":
                tool_names[_block_get(block, "id")] = _block_get(block, "name")
        raw = next((b for b in content if _block_get(b, "type") == "gemini_raw"), None)
        if raw is not None:  # replay the model's own turn unchanged
            contents.append(raw.content)
            continue
        parts: list[types.Part] = []
        for block in content:
            kind = _block_get(block, "type")
            if kind == "text":
                parts.append(types.Part(text=_block_get(block, "text")))
            elif kind == "tool_use":
                call_id = _block_get(block, "id")
                parts.append(types.Part(function_call=types.FunctionCall(
                    id=None if call_id.startswith(LOCAL_ID_PREFIX) else call_id,
                    name=_block_get(block, "name"), args=_block_get(block, "input") or {})))
            elif kind == "tool_result":
                call_id = _block_get(block, "tool_use_id")
                payload = _parse_tool_output(_block_get(block, "content"))
                response = {"error": payload} if _block_get(block, "is_error") else {"result": payload}
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=None if call_id.startswith(LOCAL_ID_PREFIX) else call_id,
                    name=tool_names.get(call_id, "tool"), response=response)))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def from_gemini_response(response: types.GenerateContentResponse, model: str = "") -> GeminiResponse:
    """Convert a Gemini reply into text / tool_use blocks plus a stop reason."""
    if not response.candidates:
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        if block_reason:
            raise LLMUnavailableError("refusal", f"the request was blocked by the model's safety filters ({block_reason})")
        raise LLMUnavailableError("empty", "the model returned an empty response")

    candidate = response.candidates[0]
    finish = getattr(candidate.finish_reason, "name", str(candidate.finish_reason or ""))
    if finish in BLOCKED_FINISH_REASONS:
        raise LLMUnavailableError("refusal", f"the model's safety filters stopped the response ({finish})")

    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    blocks: list = []
    for part in parts:
        if part.function_call:
            call = part.function_call
            blocks.append(ToolUseBlock(id=call.id or f"{LOCAL_ID_PREFIX}{uuid.uuid4().hex[:8]}",
                                       name=call.name, input=dict(call.args or {})))
        elif part.text and not part.thought:
            blocks.append(TextBlock(text=part.text))

    if not blocks and finish == "MALFORMED_FUNCTION_CALL":
        raise LLMUnavailableError("bad_response", "the model produced a malformed tool call")
    if candidate.content:
        blocks.append(RawGeminiContent(content=candidate.content, model=model))

    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    elif finish == "MAX_TOKENS":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return GeminiResponse(content=blocks, stop_reason=stop, usage=response.usage_metadata, model=model)


# --------------------------------------------------------------------------- client


class GeminiClient:
    """Gemini backend with model failover.

    Each Gemini model has its own quota. When the current model is out of quota (429) or
    overloaded (503), it is put on a cool-down and the error is marked ``retryable`` so the
    agent can restart the question on the next model in ``settings.gemini_models``.
    """

    provider = "Google Gemini"

    def __init__(self, settings: Settings, client: Any = None):
        self.models = settings.gemini_models
        self.max_tokens = settings.llm_max_tokens
        self._cooldown_until: dict[str, float] = {}
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),  # milliseconds
                # Transient server errors are retried by the SDK. 429/503 are not: they mean
                # "this model has no capacity for you", and switching model is faster.
                retry_options=types.HttpRetryOptions(attempts=settings.llm_max_retries + 1,
                                                     http_status_codes=RETRY_STATUS_CODES),
            ),
        )

    @property
    def model(self) -> str:
        """The first model that is not cooling down (the primary model if all are)."""
        now = time.monotonic()
        return next((m for m in self.models if self._cooldown_until.get(m, 0.0) <= now), self.models[0])

    def has_available_model(self) -> bool:
        now = time.monotonic()
        return any(self._cooldown_until.get(m, 0.0) <= now for m in self.models)

    def _cool_down(self, model: str, seconds: float) -> None:
        self._cooldown_until[model] = time.monotonic() + seconds
        logger.warning("Gemini model %s unavailable; cooling down for %.0fs", model, seconds)

    def create(self, *, system: str, messages: list[dict], tools: list[dict] | None = None,
               output_format: dict | None = None, effort: str | None = None) -> GeminiResponse:
        model = _conversation_model(messages) or self.model
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
            # The agent runs the tool loop itself (validation, tracing, grounding), so the
            # SDK must not execute functions automatically.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tools:
            config["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in tools
            ])]
        if output_format:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = output_format["schema"]

        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=model, contents=to_gemini_contents(messages),
                config=types.GenerateContentConfig(**config))
        except errors.ClientError as exc:
            if exc.code == 429:
                self._cool_down(model, _retry_delay_seconds(exc) or 60.0)
                raise LLMUnavailableError("rate_limit", f"the Gemini quota for {model} is used up",
                                          retryable=True) from exc
            if exc.code == 404:  # model retired or not offered to this key: skip it from now on
                self._cool_down(model, 24 * 3600)
                raise LLMUnavailableError("model", f"model '{model}' is not available to this API key",
                                          retryable=True) from exc
            raise _client_error(exc) from exc
        except errors.ServerError as exc:
            if exc.code == 503:
                self._cool_down(model, 60.0)
                raise LLMUnavailableError("overloaded", f"{model} is overloaded", retryable=True) from exc
            raise LLMUnavailableError("server", f"the Gemini API returned an error (HTTP {exc.code})") from exc
        except errors.APIError as exc:
            raise LLMUnavailableError("server", f"the Gemini API returned an error (HTTP {exc.code})") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("timeout", "the Gemini API request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError("connection", "the Gemini API could not be reached") from exc

        result = from_gemini_response(response, model)
        usage = response.usage_metadata
        logger.info("llm provider=gemini model=%s stop=%s %.0fms in=%s out=%s", model, result.stop_reason,
                    (time.perf_counter() - started) * 1000, getattr(usage, "prompt_token_count", None),
                    getattr(usage, "candidates_token_count", None))
        return result


def _conversation_model(messages: list[dict]) -> str | None:
    """A conversation that already contains Gemini turns must continue on the same model,
    because thought signatures are only valid for the model that produced them."""
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if _block_get(block, "type") == "gemini_raw" and _block_get(block, "model"):
                    return _block_get(block, "model")
    return None


def _retry_delay_seconds(exc: errors.APIError) -> float | None:
    """Read Google's RetryInfo ("retryDelay": "37s") from an error body, if present."""
    body = exc.details if isinstance(exc.details, dict) else {}
    for detail in body.get("error", {}).get("details", []) or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def _client_error(exc: errors.ClientError) -> LLMUnavailableError:
    if exc.code in (401, 403) or "API key" in str(exc.message or ""):
        return LLMUnavailableError("auth", "the Gemini API key was rejected")
    return LLMUnavailableError("bad_request", f"the Gemini API rejected the request (HTTP {exc.code})")
