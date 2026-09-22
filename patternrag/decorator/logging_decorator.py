"""
LoggingDecorator — retrieval logging cross-cutting concern.

Adds logging of query string, result count, and elapsed time to any
RetrieverStrategy without modifying it.

This is architecturally equivalent to the inline logging statements in
the monolithic pipeline, but separated into its own composable class.
"""
from __future__ import annotations

import logging
import time

from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.decorator.base import RetrieverDecorator

logger = logging.getLogger(__name__)


class LoggingDecorator(RetrieverDecorator):
    """
    Logs retrieval activity before and after delegating to the wrapped strategy.

    Logged information:
        - Query string (truncated to 80 chars for readability)
        - Number of documents returned
        - Retrieval wall-clock time in milliseconds

    Args:
        wrapped: The inner :class:`~patternrag.strategy.base.RetrieverStrategy`.
    """

    def __init__(self, wrapped: RetrieverStrategy) -> None:
        super().__init__(wrapped)

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Delegate retrieval and log the outcome.

        Args:
            query: User query string.
            k:     Number of results requested.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc` (unchanged from wrapped).
        """
        logger.debug("[PATTERNRAG] Retrieval START | query=%r | k=%d", query[:80], k)
        start = time.perf_counter()

        results = self._wrapped.retrieve(query, k)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "[PATTERNRAG] Retrieval DONE  | docs=%d | elapsed=%.1f ms",
            len(results),
            elapsed_ms,
        )
        return results
