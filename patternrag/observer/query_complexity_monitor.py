"""
QueryComplexityMonitor — tracks query string length and outlier statistics.

Observes QUERY_RECEIVED events from the PatternRAG pipeline side-channel.
Tracks mean, min, max query character lengths and counts outlier queries
whose length is < 15 or > 200 characters.
"""
from __future__ import annotations

import logging
from typing import Any

from patternrag.observer.base import PipelineEvent, PipelineObserver

logger = logging.getLogger(__name__)


class QueryComplexityMonitor(PipelineObserver):
    """
    Observer that tracks query length distribution and outlier queries.

    Listens for ``QUERY_RECEIVED`` events. Outlier criteria:
        ``len(query) < 15`` OR ``len(query) > 200``
    """

    def __init__(self) -> None:
        self._query_lengths: list[int] = []
        self._outlier_query_count: int = 0

    def on_event(self, event: PipelineEvent) -> None:
        """
        Record query length and outlier status on QUERY_RECEIVED events.

        Guaranteed never to raise unhandled exceptions.
        """
        try:
            if event.event_type == "QUERY_RECEIVED":
                query = event.payload.get("query", "")
                q_len = len(str(query))
                self._query_lengths.append(q_len)
                if q_len < 15 or q_len > 200:
                    self._outlier_query_count += 1
                logger.debug(
                    "[QueryComplexityMonitor] query_len=%d, outlier=%s",
                    q_len,
                    q_len < 15 or q_len > 200,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[QueryComplexityMonitor] Exception handling event: %s", exc)

    def get_summary(self) -> dict[str, Any]:
        """
        Return query complexity summary statistics.

        Returns:
            Dict with keys:
                - ``mean_query_length``: float
                - ``min_query_length``: int
                - ``max_query_length``: int
                - ``outlier_query_count``: int
        """
        if not self._query_lengths:
            return {
                "mean_query_length": 0.0,
                "min_query_length": 0,
                "max_query_length": 0,
                "outlier_query_count": 0,
            }

        return {
            "mean_query_length": sum(self._query_lengths) / len(self._query_lengths),
            "min_query_length": min(self._query_lengths),
            "max_query_length": max(self._query_lengths),
            "outlier_query_count": self._outlier_query_count,
        }

    @property
    def query_lengths(self) -> list[int]:
        """All recorded query lengths in arrival order."""
        return list(self._query_lengths)

    @property
    def outlier_query_count(self) -> int:
        """Number of queries identified as outliers."""
        return self._outlier_query_count
