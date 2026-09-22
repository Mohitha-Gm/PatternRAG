"""
LatencyLogger observer — records retrieval and generation latency.

Receives RETRIEVAL_DONE and GENERATION_DONE events and accumulates
per-query and aggregate latency statistics.
"""
from __future__ import annotations

import logging
from typing import Any

from patternrag.observer.base import PipelineEvent, PipelineObserver

logger = logging.getLogger(__name__)


class LatencyLogger(PipelineObserver):
    """
    Observer that records retrieval and generation latency from pipeline events.

    Accumulated data is accessible via :meth:`get_summary` or
    per-query via :attr:`retrieval_latencies` / :attr:`generation_latencies`.
    """

    def __init__(self) -> None:
        self.retrieval_latencies: list[float] = []
        self.generation_latencies: list[float] = []

    def on_event(self, event: PipelineEvent) -> None:
        if event.event_type == "RETRIEVAL_DONE":
            ms = event.payload.get("elapsed_ms", 0.0)
            self.retrieval_latencies.append(float(ms))
            logger.debug("[LatencyLogger] Retrieval %.1f ms", ms)

        elif event.event_type == "GENERATION_DONE":
            ms = event.payload.get("elapsed_ms", 0.0)
            self.generation_latencies.append(float(ms))
            logger.debug("[LatencyLogger] Generation %.1f ms", ms)

    def get_summary(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0}
            import statistics
            sorted_v = sorted(values)
            p95_idx = max(0, int(len(sorted_v) * 0.95) - 1)
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": sorted_v[p95_idx],
            }

        return {
            "retrieval_latency_ms": stats(self.retrieval_latencies),
            "generation_latency_ms": stats(self.generation_latencies),
            "n_queries": len(self.retrieval_latencies),
        }
