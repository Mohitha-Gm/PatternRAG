"""
MetricsCollector observer — accumulates per-query retrieval metrics.

Receives RETRIEVAL_DONE events and stores the document IDs retrieved for
each query, enabling post-hoc retrieval quality evaluation.
"""
from __future__ import annotations

import logging
from typing import Any

from patternrag.observer.base import PipelineEvent, PipelineObserver

logger = logging.getLogger(__name__)


class MetricsCollector(PipelineObserver):
    """
    Observer that accumulates retrieval data for post-hoc evaluation.

    Stores:
        - The list of retrieved document IDs per query (for metric computation).
        - The query string for each event.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def on_event(self, event: PipelineEvent) -> None:
        if event.event_type == "RETRIEVAL_DONE":
            doc_ids = event.payload.get("retrieved_ids", [])
            query = event.payload.get("query", "")
            self._records.append({
                "query": query,
                "retrieved_ids": doc_ids,
                "n_docs": len(doc_ids),
                "timestamp": event.timestamp,
            })
            logger.debug(
                "[MetricsCollector] Recorded %d docs for query %r",
                len(doc_ids),
                str(query)[:60],
            )

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_queries_tracked": len(self._records),
            "mean_docs_retrieved": (
                sum(r["n_docs"] for r in self._records) / len(self._records)
                if self._records else 0.0
            ),
        }

    @property
    def records(self) -> list[dict[str, Any]]:
        """All accumulated retrieval records."""
        return list(self._records)
