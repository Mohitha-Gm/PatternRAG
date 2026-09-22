"""
GoF Factory Pattern — PipelineFactory.

Responsible for constructing and wiring the complete PatternRAG pipeline
from configuration.  This is a CONSTRUCTION-TIME operation only.

The factory is called ONCE before any queries are processed.
It is NOT invoked during query execution.

Factory responsibilities:
    1. Read experiment config  (experiment.yaml)
    2. Read PatternRAG config  (patternrag.yaml)
    3. Instantiate shared infrastructure (indexes, embedder)
    4. Instantiate the configured RetrieverStrategy
    5. Apply decorators in declared order
    6. Instantiate observers
    7. Create EventDispatcher and register observers
    8. Instantiate the LLM generator
    9. Assemble and return PatternRAGPipeline

The caller (run_patternrag.py) passes pre-loaded indexes and embedder
so that both systems use the same index objects — a required condition
for the fair comparison.
"""
from __future__ import annotations

from typing import Any

from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.llm.generator import LLMGenerator

from patternrag.strategy.base import RetrieverStrategy
from patternrag.strategy.bm25_retriever import BM25Retriever
from patternrag.strategy.dense_retriever import DenseRetriever
from patternrag.strategy.hybrid_retriever import HybridRetriever

from patternrag.decorator.base import RetrieverDecorator
from patternrag.decorator.caching_decorator import CachingDecorator
from patternrag.decorator.logging_decorator import LoggingDecorator

from patternrag.observer.base import EventDispatcher, PipelineObserver
from patternrag.observer.latency_logger import LatencyLogger
from patternrag.observer.metrics_collector import MetricsCollector
from patternrag.observer.cost_monitor import CostMonitor


class PipelineFactory:
    """
    Constructs a fully assembled :class:`~patternrag.pipeline.PatternRAGPipeline`
    from experiment and pattern configuration dicts.

    Usage::

        from shared.config import load_yaml
        exp_cfg = load_yaml("configs/experiment.yaml")
        pat_cfg = load_yaml("configs/patternrag.yaml")

        factory = PipelineFactory()
        pipeline = factory.build(
            exp_cfg=exp_cfg,
            pat_cfg=pat_cfg,
            bm25_index=bm25_index,
            faiss_index=faiss_index,
            embedder=embedder,
        )
    """

    # ------------------------------------------------------------------
    # Strategy registry
    # ------------------------------------------------------------------

    _RETRIEVER_TYPES: dict[str, type] = {
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
    }

    # ------------------------------------------------------------------
    # Decorator registry
    # ------------------------------------------------------------------

    _DECORATOR_TYPES: dict[str, type] = {
        "caching": CachingDecorator,
        "logging": LoggingDecorator,
    }

    # ------------------------------------------------------------------
    # Observer registry
    # ------------------------------------------------------------------

    _OBSERVER_TYPES: dict[str, type] = {
        "latency_logger": LatencyLogger,
        "metrics_collector": MetricsCollector,
        "cost_monitor": CostMonitor,
    }

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        exp_cfg: dict[str, Any],
        pat_cfg: dict[str, Any],
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        embedder: SentenceTransformerEmbedder,
    ) -> "patternrag.pipeline.PatternRAGPipeline":  # type: ignore[name-defined]
        """
        Assemble and return a fully wired PatternRAGPipeline.

        Args:
            exp_cfg:     Experiment config dict (from ``experiment.yaml``).
            pat_cfg:     PatternRAG config dict (from ``patternrag.yaml``).
            bm25_index:  Pre-loaded :class:`~shared.indexing.bm25_index.BM25Index`.
            faiss_index: Pre-loaded :class:`~shared.indexing.faiss_index.FAISSIndex`.
            embedder:    Pre-loaded :class:`~shared.embedding.embedder.SentenceTransformerEmbedder`.

        Returns:
            A ready-to-use :class:`~patternrag.pipeline.PatternRAGPipeline`.
        """
        # ── Step 1: Instantiate retriever strategy ─────────────────────
        retriever = self._build_retriever(pat_cfg, bm25_index, faiss_index, embedder, exp_cfg)

        # ── Step 2: Apply decorators ───────────────────────────────────
        retriever = self._apply_decorators(pat_cfg, retriever)

        # ── Step 3: Build observers and dispatcher ─────────────────────
        dispatcher = EventDispatcher()
        self._register_observers(pat_cfg, dispatcher)

        # ── Step 4: Instantiate LLM generator ─────────────────────────
        llm_cfg = exp_cfg.get("llm", {})
        generator = LLMGenerator(
            model_name=llm_cfg.get("model_name", "openai/gpt-oss-120b"),
            max_tokens=llm_cfg.get("max_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.0),
            max_retries=llm_cfg.get("max_retries", 5),
            initial_backoff=llm_cfg.get("initial_backoff", 2.0),
        )

        # ── Step 5: Assemble pipeline ──────────────────────────────────
        from patternrag.pipeline import PatternRAGPipeline  # late import to avoid circularity
        return PatternRAGPipeline(
            retriever=retriever,
            generator=generator,
            dispatcher=dispatcher,
            config=exp_cfg,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_retriever(
        self,
        pat_cfg: dict[str, Any],
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        embedder: SentenceTransformerEmbedder,
        exp_cfg: dict[str, Any],
    ) -> RetrieverStrategy:
        retriever_cfg = pat_cfg.get("retriever", {})
        retriever_type = retriever_cfg.get("type", "hybrid").lower()

        rrf_k = exp_cfg.get("retrieval", {}).get("rrf_k", 60)

        if retriever_type == "bm25":
            return BM25Retriever(bm25_index)

        elif retriever_type == "dense":
            return DenseRetriever(faiss_index, embedder)

        elif retriever_type == "hybrid":
            bm25_r = BM25Retriever(bm25_index)
            dense_r = DenseRetriever(faiss_index, embedder)
            return HybridRetriever(bm25_r, dense_r, rrf_k=rrf_k)

        else:
            available = ", ".join(self._RETRIEVER_TYPES.keys())
            raise ValueError(
                f"Unknown retriever type: {retriever_type!r}. "
                f"Available: {available}"
            )

    def _apply_decorators(
        self,
        pat_cfg: dict[str, Any],
        retriever: RetrieverStrategy,
    ) -> RetrieverStrategy:
        """
        Wrap *retriever* with declared decorators.

        The config lists decorators innermost-first; the factory wraps them
        such that the last declared becomes the outermost (first called).
        """
        decorator_cfgs: list[dict] = pat_cfg.get("decorators", [])
        # Innermost first: wrap from left to right so last = outermost.
        wrapped = retriever
        for dec_cfg in decorator_cfgs:
            dec_type = dec_cfg.get("type", "").lower()
            dec_class = self._DECORATOR_TYPES.get(dec_type)
            if dec_class is None:
                available = ", ".join(self._DECORATOR_TYPES.keys())
                raise ValueError(
                    f"Unknown decorator type: {dec_type!r}. Available: {available}"
                )
            wrapped = dec_class(wrapped)
        return wrapped

    def _register_observers(
        self,
        pat_cfg: dict[str, Any],
        dispatcher: EventDispatcher,
    ) -> None:
        observer_cfgs: list[dict] = pat_cfg.get("observers", [])
        for obs_cfg in observer_cfgs:
            obs_type = obs_cfg.get("type", "").lower()
            obs_class = self._OBSERVER_TYPES.get(obs_type)
            if obs_class is None:
                available = ", ".join(self._OBSERVER_TYPES.keys())
                raise ValueError(
                    f"Unknown observer type: {obs_type!r}. Available: {available}"
                )
            dispatcher.register(obs_class())
