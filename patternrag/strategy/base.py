"""
GoF Strategy Pattern — RetrieverStrategy interface.

Defines the common interface that all concrete retrieval strategies must
implement.  This decouples the pipeline from concrete retrieval
implementations, enabling them to be swapped via configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from shared.types import ScoredDoc


class RetrieverStrategy(ABC):
    """
    Abstract base class for all retrieval strategies.

    Concrete implementations:
        - :class:`~patternrag.strategy.bm25_retriever.BM25Retriever`
        - :class:`~patternrag.strategy.dense_retriever.DenseRetriever`
        - :class:`~patternrag.strategy.hybrid_retriever.HybridRetriever`

    The Decorator base class also extends this interface, allowing decorators
    to be composed transparently with concrete strategies.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Retrieve the top-*k* documents relevant to *query*.

        Args:
            query: The user query string.
            k:     Number of documents to return.

        Returns:
            List of :class:`~shared.types.ScoredDoc` ordered by relevance
            (highest score first).
        """
        ...
