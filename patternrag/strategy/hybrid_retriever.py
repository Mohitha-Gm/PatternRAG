"""
HybridRetriever — concrete Strategy for hybrid BM25+dense retrieval.

Implements :class:`~patternrag.strategy.base.RetrieverStrategy` by
composing :class:`BM25Retriever` and :class:`DenseRetriever` using
standard Reciprocal Rank Fusion (no alpha weight).

This is IDENTICAL to the RRF merge logic in ``monolithic/pipeline.py``.
The retrieval equivalence test verifies that both implementations return
the same document ordering for the same query and configuration.

RRF formula:
    score(d) = Σ_i  1 / (rrf_k + rank_i(d))

where rrf_k is the standard RRF constant (default 60) and rank_i is the
1-based position of document d in ranked list i.
"""
from __future__ import annotations

from shared.types import Document, ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.strategy.bm25_retriever import BM25Retriever
from patternrag.strategy.dense_retriever import DenseRetriever


class HybridRetriever(RetrieverStrategy):
    """
    Hybrid retrieval strategy that fuses BM25 and dense results via RRF.

    Args:
        bm25_retriever:  A :class:`BM25Retriever` instance.
        dense_retriever: A :class:`DenseRetriever` instance.
        rrf_k:           RRF constant (default 60, per standard literature).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Retrieve top-*k* documents by fusing BM25 and dense results with RRF.

        Args:
            query: Raw query string.
            k:     Number of results to return.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc` fused by RRF.
        """
        bm25_results = self._bm25.retrieve(query, k)
        dense_results = self._dense.retrieve(query, k)
        return self._rrf_merge(bm25_results, dense_results, k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        bm25_results: list[ScoredDoc],
        dense_results: list[ScoredDoc],
        k: int,
    ) -> list[ScoredDoc]:
        """
        Standard Reciprocal Rank Fusion merge.

        This is algorithmically identical to MonolithicRAGPipeline._rrf_merge().
        """
        rrf_k = self._rrf_k
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, sd in enumerate(bm25_results, start=1):
            doc_id = sd.doc.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[doc_id] = sd.doc

        for rank, sd in enumerate(dense_results, start=1):
            doc_id = sd.doc.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[doc_id] = sd.doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            ScoredDoc(doc=doc_map[doc_id], score=rrf_score)
            for doc_id, rrf_score in ranked[:k]
        ]
