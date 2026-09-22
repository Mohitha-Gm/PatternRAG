"""
TFIDFRetriever — concrete Strategy for TF-IDF keyword retrieval.

Implements :class:`~patternrag.strategy.base.RetrieverStrategy` using
the shared :class:`~shared.indexing.tfidf_index.TFIDFIndex`.
"""
from __future__ import annotations

from shared.indexing.tfidf_index import TFIDFIndex
from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy


class TFIDFRetriever(RetrieverStrategy):
    """
    TF-IDF keyword retrieval strategy.

    Args:
        tfidf_index: A pre-loaded :class:`~shared.indexing.tfidf_index.TFIDFIndex`.
    """

    def __init__(self, tfidf_index: TFIDFIndex) -> None:
        self._tfidf_index = tfidf_index

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Retrieve top-*k* documents using TF-IDF cosine similarity.

        Args:
            query: Raw query string.
            k:     Number of results.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc`.
        """
        return self._tfidf_index.search(query, k)
