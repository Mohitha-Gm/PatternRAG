"""
LLM generator — single configured language model using Groq.

Both the monolithic pipeline and PatternRAG use this class.
Using the same model with the same parameters is required for a fair
functional comparison.

The model is configured via experiment.yaml (``llm.model_name``).
The API key is read from the ``GROQ_API_KEY`` environment variable.

If the API key is not set, generation will raise a clear error rather
than silently falling back to another model.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from groq import Groq, RateLimitError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract retry-after seconds from Groq exception headers or message."""
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        retry_val = response.headers.get("retry-after")
        if retry_val:
            try:
                return float(retry_val)
            except (ValueError, TypeError):
                pass

    msg = str(exc)
    match = re.search(r"try again in ([\d\.]+)s", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass
    return None


class LLMGenerator:
    """
    Wraps the Groq chat completion API for RAG answer generation.

    Args:
        model_name:      Groq model identifier, e.g. ``"openai/gpt-oss-120b"``.
        max_tokens:      Maximum number of tokens in the generated answer.
        temperature:     Sampling temperature (0.0 = deterministic greedy).
        max_retries:     Maximum application-level retries on rate limits / connection errors.
        initial_backoff: Initial backoff delay in seconds if server specifies no Retry-After.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key, "
                "then re-run the script."
            )
        self._client = Groq(api_key=api_key, max_retries=2, timeout=60.0)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.initial_backoff = max(0.1, initial_backoff)

    def generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """
        Send *prompt* to the LLM and return the answer along with usage info.
        Retries on RateLimitError (429) and transient connection/server errors
        up to `max_retries` attempts, respecting server Retry-After headers when available.

        Args:
            prompt: The full prompt string (system + user messages combined).

        Returns:
            A tuple ``(answer_text, usage_dict)`` where ``usage_dict``
            contains ``{"prompt_tokens": int, "completion_tokens": int}``.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful question-answering assistant. "
                    "Answer the question using only the provided context. "
                    "Be concise and factual."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw_content = response.choices[0].message.content
                answer = (raw_content or "").strip()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
                return answer, usage

            except RateLimitError as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "Groq rate limit exceeded after %d attempts for model %s: %s",
                        attempt, self.model_name, exc,
                    )
                    raise

                retry_after = _extract_retry_after(exc)
                if retry_after is not None:
                    wait_time = max(retry_after + 0.5, self.initial_backoff)
                else:
                    wait_time = self.initial_backoff * (2 ** (attempt - 1))

                logger.warning(
                    "Groq RateLimitError (429) on attempt %d/%d. Waiting %.2fs before retry... (error: %s)",
                    attempt, self.max_retries, wait_time, exc,
                )
                time.sleep(wait_time)

            except (APIConnectionError, InternalServerError) as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "Groq transient error (%s) after %d attempts for model %s: %s",
                        type(exc).__name__, attempt, self.model_name, exc,
                    )
                    raise

                wait_time = self.initial_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Groq transient error (%s) on attempt %d/%d: %s. Waiting %.2fs before retry...",
                    type(exc).__name__, attempt, self.max_retries, exc, wait_time,
                )
                time.sleep(wait_time)

            except Exception as exc:
                logger.error("Groq API generation failed for model %s: %s", self.model_name, exc)
                raise
