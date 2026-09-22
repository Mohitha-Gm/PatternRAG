"""
ResultFilterDecorator — post-retrieval score threshold filtering.

Adds score-based filtering to any RetrieverStrategy without modifying it.
Retains documents satisfying doc.score >= score_threshold. If all documents
fall below the threshold, retains the top-1 document so that downstream
generation never receives an empty context.

Part of the GoF Decorator hierarchy extending RetrieverDecorator.
"""
from __future__ import annotations

from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.decorator.base import RetrieverDecorator


class ResultFilterDecorator(RetrieverDecorator):
    """
    Filters retrieved documents by a relevance-score threshold.

    Args:
        wrapped: The inner :class:`~patternrag.strategy.base.RetrieverStrategy`.
        score_threshold: Minimum score required to retain a document. Defaults to 0.0.
    """

    def __init__(
        self,
        wrapped: RetrieverStrategy,
        score_threshold: float = 0.0,
    ) -> None:
        super().__init__(wrapped)
        self._score_threshold = float(score_threshold)

    @property
    def score_threshold(self) -> float:
        """The active score threshold."""
        return self._score_threshold

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Delegate retrieval to the wrapped strategy and filter results.

        Retains documents with score >= score_threshold.
        If all documents fall below the threshold, retains the top-1 document.
        """
        scored_docs = self._wrapped.retrieve(query, k)
        if not scored_docs:
            return []

        filtered = [sd for sd in scored_docs if sd.score >= self._score_threshold]
        if not filtered:
            return [scored_docs[0]]
        return filtered
