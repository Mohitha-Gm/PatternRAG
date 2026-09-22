"""
CostMonitor observer — tracks token usage and estimated API cost.

Receives GENERATION_DONE events and accumulates prompt and completion
token counts.  Estimates cost based on the configured model's pricing.
"""
from __future__ import annotations

import logging
from typing import Any

from patternrag.observer.base import PipelineEvent, PipelineObserver

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (USD) for Groq llama-3.3-70b-versatile.
# Update these if model pricing changes.
_COST_PER_1M_PROMPT_TOKENS = 0.59
_COST_PER_1M_COMPLETION_TOKENS = 0.79


class CostMonitor(PipelineObserver):
    """
    Observer that tracks LLM token usage and estimates API cost.

    Accumulates prompt and completion tokens from GENERATION_DONE events.
    Cost estimate is based on Groq ``llama-3.3-70b-versatile`` pricing;
    update the module constants if using a different model.
    """

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._n_generations: int = 0

    def on_event(self, event: PipelineEvent) -> None:
        if event.event_type == "GENERATION_DONE":
            pt = event.payload.get("prompt_tokens", 0)
            ct = event.payload.get("completion_tokens", 0)
            self._prompt_tokens += int(pt)
            self._completion_tokens += int(ct)
            self._n_generations += 1
            logger.debug(
                "[CostMonitor] Generation %d | prompt=%d | completion=%d",
                self._n_generations, pt, ct,
            )

    def get_summary(self) -> dict[str, Any]:
        estimated_cost = (
            self._prompt_tokens / 1_000_000 * _COST_PER_1M_PROMPT_TOKENS
            + self._completion_tokens / 1_000_000 * _COST_PER_1M_COMPLETION_TOKENS
        )
        return {
            "total_prompt_tokens": self._prompt_tokens,
            "total_completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "n_generations": self._n_generations,
            "estimated_cost_usd": round(estimated_cost, 6),
        }
