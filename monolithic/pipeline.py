"""
Monolithic RAG Pipeline — Baseline implementation.

============================================================
ARCHITECTURAL NOTE
============================================================
This file is intentionally monolithic.  All pipeline logic —
retrieval mode selection, BM25 search, dense search, hybrid
Reciprocal Rank Fusion, in-memory caching, logging, latency
tracking, prompt construction, LLM generation, and basic
monitoring — lives inline in this single class.

Do NOT refactor this file to introduce:
  - Abstract base classes for retrieval
  - Decorator classes for caching or logging
  - Observer or event-bus classes for monitoring
  - A factory for pipeline construction

The monolithic structure is the independent variable in the
research comparison.  Maintaining this structure is required
for the study to be valid.

The pipeline calls shared infrastructure (indexes, embedder,
prompt, LLM generator) because those are controlled variables
— not because it is pattern-based.
============================================================
"""
from __future__ import annotations

import logging
import time
from typing import Any

from shared.types import Document, ScoredDoc
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.llm.generator import LLMGenerator
from shared.prompts.templates import build_prompt

logger = logging.getLogger(__name__)


class MonolithicRAGPipeline:
    """
    A competent but intentionally monolithic RAG pipeline.

    All concerns — retrieval, caching, logging, latency tracking,
    monitoring, and generation — are handled inline within :meth:`run`.

    Args:
        bm25_index:  Pre-loaded :class:`~shared.indexing.bm25_index.BM25Index`.
        faiss_index: Pre-loaded :class:`~shared.indexing.faiss_index.FAISSIndex`.
        embedder:    :class:`~shared.embedding.embedder.SentenceTransformerEmbedder`.
        generator:   :class:`~shared.llm.generator.LLMGenerator`.
        config:      Experiment config dict (from ``experiment.yaml``).
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        embedder: SentenceTransformerEmbedder,
        generator: LLMGenerator,
        config: dict[str, Any],
    ) -> None:
        self._bm25_index = bm25_index
        self._faiss_index = faiss_index
        self._embedder = embedder
        self._generator = generator
        self._config = config

        # In-memory result cache: maps query string → run result dict.
        # Plain dict — intentionally no abstraction.
        self._cache: dict[str, dict[str, Any]] = {}

        # In-memory monitoring counters.
        self._total_queries: int = 0
        self._cache_hits: int = 0
        self._total_retrieval_ms: float = 0.0
        self._total_generation_ms: float = 0.0

        # Retrieval mode: "bm25" | "dense" | "hybrid"
        # In the research experiments both systems always use "hybrid".
        self._retrieval_mode: str = config.get("retrieval_mode", "hybrid")
        self._top_k: int = config.get("retrieval", {}).get("top_k", 10)
        self._rrf_k: int = config.get("retrieval", {}).get("rrf_k", 60)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str, k: int | None = None) -> dict[str, Any]:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query: The user question string.
            k:     Number of documents to retrieve.  Defaults to ``top_k``
                   from the experiment config.

        Returns:
            Dict with keys:
              - ``answer``          : str
              - ``retrieved_docs``  : list[ScoredDoc]
              - ``retrieved_ids``   : list[str]
              - ``retrieval_ms``    : float
              - ``generation_ms``   : float
              - ``total_ms``        : float
              - ``prompt_tokens``   : int
              - ``completion_tokens``: int
              - ``cache_hit``       : bool
        """
        if k is None:
            k = self._top_k

        self._total_queries += 1
        pipeline_start = time.perf_counter()

        # ── Cache check ───────────────────────────────────────────────
        cache_key = f"{query}||k={k}"
        if cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            logger.debug("[MONOLITHIC] Cache HIT for query: %r", query)
            return {**cached, "cache_hit": True}

        # ── Retrieval ─────────────────────────────────────────────────
        retrieval_start = time.perf_counter()

        if self._retrieval_mode == "bm25":
            scored_docs = self._bm25_index.search(query, k)

        elif self._retrieval_mode == "dense":
            query_vec = self._embedder.encode(query)
            scored_docs = self._faiss_index.search(query_vec, k)

        else:  # hybrid (default)
            # BM25 branch
            bm25_results = self._bm25_index.search(query, k)

            # Dense branch
            query_vec = self._embedder.encode(query)
            dense_results = self._faiss_index.search(query_vec, k)

            # Reciprocal Rank Fusion (standard, no alpha weight)
            # score(d) = Σ  1 / (rrf_k + rank_i(d))
            scored_docs = self._rrf_merge(bm25_results, dense_results, k)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        self._total_retrieval_ms += retrieval_ms

        retrieved_ids = [sd.doc.id for sd in scored_docs]
        logger.debug(
            "[MONOLITHIC] Query: %r | Retrieved %d docs in %.1f ms",
            query[:80],
            len(scored_docs),
            retrieval_ms,
        )

        # ── Prompt construction ───────────────────────────────────────
        prompt = build_prompt(query, scored_docs)

        # ── LLM generation ────────────────────────────────────────────
        generation_start = time.perf_counter()
        answer, usage = self._generator.generate(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        self._total_generation_ms += generation_ms

        logger.debug(
            "[MONOLITHIC] Generated answer in %.1f ms | tokens: %d+%d",
            generation_ms,
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        result = {
            "answer": answer,
            "retrieved_docs": scored_docs,
            "retrieved_ids": retrieved_ids,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cache_hit": False,
        }

        # ── Cache store ───────────────────────────────────────────────
        self._cache[cache_key] = result

        return result

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
        Merge BM25 and dense results using standard Reciprocal Rank Fusion.

        RRF score for document d:
            score(d) = Σ_i  1 / (rrf_k + rank_i(d))

        where rrf_k is the RRF constant (default 60) and rank_i is the
        1-based rank of d in result list i (0 if absent).

        Args:
            bm25_results:  Ranked BM25 results.
            dense_results: Ranked dense results.
            k:             Number of results to return.

        Returns:
            Top-k documents by combined RRF score.
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

    # ------------------------------------------------------------------
    # Monitoring summary
    # ------------------------------------------------------------------

    def get_monitoring_summary(self) -> dict[str, Any]:
        """
        Return basic monitoring counters accumulated during this session.

        Returns:
            Dict with query count, cache hit rate, and mean latencies.
        """
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
            "mean_retrieval_ms": (
                self._total_retrieval_ms / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
            "mean_generation_ms": (
                self._total_generation_ms / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
        }
