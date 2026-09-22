"""
BM25Retriever — concrete Strategy for keyword-based retrieval.

Implements :class:`~patternrag.strategy.base.RetrieverStrategy` using
the shared :class:`~shared.indexing.bm25_index.BM25Index`.

The underlying BM25 algorithm is identical to what the monolithic pipeline
calls inline.  The only difference is architectural: here the logic is
encapsulated in a dedicated class conforming to RetrieverStrategy.
"""
from __future__ import annotations

from shared.indexing.bm25_index import BM25Index
from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy


class BM25Retriever(RetrieverStrategy):
    """
    BM25 keyword retrieval strategy.

    Args:
        bm25_index: A pre-loaded :class:`~shared.indexing.bm25_index.BM25Index`.
    """

    def __init__(self, bm25_index: BM25Index) -> None:
        self._bm25_index = bm25_index

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Retrieve top-*k* documents using BM25 scoring.

        Args:
            query: Raw query string.
            k:     Number of results.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc`.
        """
        return self._bm25_index.search(query, k)
