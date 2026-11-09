"""Claude (Anthropic) client wrapper, plus the provider interface shared with the Gemini backend.

Model/provider: Anthropic Claude, default ``claude-opus-5-5`` (configurable via ANTHROPIC_MODEL).
Used for: (1) understanding the question and choosing tools in the agent loop,
(2) writing the final grounded answer, (3) rephrasing dashboard insights.

Every SDK failure is translated into :class:`LLMUnavailableError` with a short,
user-presentable reason; the agent then falls back to the rule-based engine.
The SDK itself already retries 408/409/429/5xx and connection errors with
exponential backoff (``max_retries``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import anthropic

from app.config import Settings

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailableError(Exception):
    """The LLM could not produce a response. ``retryable`` means another model may succeed."""

    def __init__(self, kind: str, reason: str, retryable: bool = False):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.retryable = retryable


class LLMBackend(Protocol):
    """What the agent needs from any LLM provider (implemented by ClaudeClient and GeminiClient)."""

    provider: str
    model: str

    def create(self, *, system: str, messages: list[dict], tools: list[dict] | None = None,
               output_format: dict | None = None, effort: str | None = None) -> Any: ...


class ClaudeClient:
    provider = "Anthropic Claude"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.model = settings.anthropic_model
        self.effort = settings.llm_effort
        self.max_tokens = settings.llm_max_tokens
        self._use_fallbacks = settings.llm_refusal_fallback
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def create(self, *, system: str, messages: list[dict], tools: list[dict] | None = None,
               output_format: dict | None = None, effort: str | None = None) -> Any:
        output_config: dict[str, Any] = {"effort": effort or self.effort}
        if output_format:
            output_config["format"] = output_format
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": output_config,
            # Caches the stable prefix (tools + system prompt) across requests.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = tools

        started = time.perf_counter()
        try:
            response = self._send(kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailableError("auth", "the Anthropic API key was rejected") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMUnavailableError("permission", "the API key is not allowed to use this model") from exc
        except anthropic.NotFoundError as exc:
            raise LLMUnavailableError("model", f"model '{self.model}' was not found or is not available") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("rate_limit", "the API rate limit was exceeded") from exc
        except anthropic.BadRequestError as exc:
            raise LLMUnavailableError("bad_request", f"the API rejected the request ({exc.message})") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError("server", f"the API returned an error (HTTP {exc.status_code})") from exc
        except anthropic.APITimeoutError as exc:  # subclass of APIConnectionError: catch first
            raise LLMUnavailableError("timeout", "the API request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("connection", "the API could not be reached") from exc
        except anthropic.AnthropicError as exc:
            raise LLMUnavailableError("client", f"the LLM client failed ({type(exc).__name__})") from exc

        usage = getattr(response, "usage", None)
        logger.info(
            "llm model=%s stop=%s %.0fms in=%s out=%s cache_read=%s",
            self.model, response.stop_reason, (time.perf_counter() - started) * 1000,
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
        )
        if response.stop_reason == "refusal":
            raise LLMUnavailableError("refusal", "the model declined to answer this request")
        return response

    def _send(self, kwargs: dict[str, Any]) -> Any:
        if self._use_fallbacks:
            try:
                return self._client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            except anthropic.BadRequestError as exc:
                # The fallback beta may not be enabled for every organisation. In that case
                # disable it for this process and resend the same request without it.
                if "anthropic-beta" not in exc.message and "fallbacks" not in exc.message:
                    raise
                logger.warning("Refusal-fallback beta unavailable (%s); continuing without it", exc.message)
                self._use_fallbacks = False
        return self._client.messages.create(**kwargs)
