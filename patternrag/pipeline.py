"""
PatternRAG Pipeline — thin runtime orchestrator.

This class is deliberately thin.  It delegates all concerns to the
assembled components:

    - Retrieval:  to the decorated RetrieverStrategy
    - Monitoring: to observers via EventDispatcher (side-channel)
    - Prompting:  to shared.prompts.templates.build_prompt
    - Generation: to shared.llm.generator.LLMGenerator

No retrieval logic, caching logic, or monitoring logic lives here.
Those concerns are handled by the Strategy+Decorator and Observer layers
respectively, constructed by PipelineFactory before any query runs.
"""
from __future__ import annotations

import time
from typing import Any

from shared.prompts.templates import build_prompt
from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.observer.base import EventDispatcher, PipelineEvent


class PatternRAGPipeline:
    """
    Thin orchestrator for the PatternRAG system.

    Args:
        retriever:   A :class:`~patternrag.strategy.base.RetrieverStrategy`
                     (possibly decorated) returned by PipelineFactory.
        generator:   A :class:`~shared.llm.generator.LLMGenerator`.
        dispatcher:  The :class:`~patternrag.observer.base.EventDispatcher`
                     with all observers already registered.
        config:      Experiment config dict (for top_k).
    """

    def __init__(
        self,
        retriever: RetrieverStrategy,
        generator,  # LLMGenerator — avoid circular import
        dispatcher: EventDispatcher,
        config: dict[str, Any],
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._dispatcher = dispatcher
        self._top_k: int = config.get("retrieval", {}).get("top_k", 10)

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
            Dict with keys matching the monolithic pipeline's return schema:
              - ``answer``           : str
              - ``retrieved_docs``   : list[ScoredDoc]
              - ``retrieved_ids``    : list[str]
              - ``retrieval_ms``     : float
              - ``generation_ms``    : float
              - ``total_ms``         : float
              - ``prompt_tokens``    : int
              - ``completion_tokens``: int
              - ``cache_hit``        : bool  (set to False; CachingDecorator
                                              handles caching transparently)
        """
        if k is None:
            k = self._top_k

        pipeline_start = time.perf_counter()

        # ── Event: QUERY_RECEIVED ─────────────────────────────────────
        self._dispatcher.notify(PipelineEvent(
            event_type="QUERY_RECEIVED",
            payload={"query": query, "k": k},
        ))

        # ── Retrieval (via decorated strategy) ────────────────────────
        retrieval_start = time.perf_counter()
        scored_docs: list[ScoredDoc] = self._retriever.retrieve(query, k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        retrieved_ids = [sd.doc.id for sd in scored_docs]

        # ── Event: RETRIEVAL_DONE ─────────────────────────────────────
        self._dispatcher.notify(PipelineEvent(
            event_type="RETRIEVAL_DONE",
            payload={
                "query": query,
                "retrieved_ids": retrieved_ids,
                "n_docs": len(scored_docs),
                "elapsed_ms": retrieval_ms,
            },
        ))

        # ── Prompt construction ───────────────────────────────────────
        prompt = build_prompt(query, scored_docs)

        # ── LLM generation ────────────────────────────────────────────
        generation_start = time.perf_counter()
        answer, usage = self._generator.generate(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # ── Event: GENERATION_DONE ────────────────────────────────────
        self._dispatcher.notify(PipelineEvent(
            event_type="GENERATION_DONE",
            payload={
                "answer": answer,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "elapsed_ms": generation_ms,
            },
        ))

        return {
            "answer": answer,
            "retrieved_docs": scored_docs,
            "retrieved_ids": retrieved_ids,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cache_hit": False,  # CachingDecorator handles this transparently
        }

    # ------------------------------------------------------------------
    # Observer access
    # ------------------------------------------------------------------

    def get_observer_summaries(self) -> list[dict[str, Any]]:
        """Return summary dicts from all registered observers."""
        summaries = []
        for observer in self._dispatcher._observers:
            summaries.append({
                "observer": type(observer).__name__,
                "summary": observer.get_summary(),
            })
        return summaries

    def get_event_log(self) -> list[dict[str, Any]]:
        """Return list of all dispatched events as serializable dicts."""
        return [
            {
                "event_type": e.event_type,
                "payload": e.payload,
                "timestamp": e.timestamp,
            }
            for e in self._dispatcher.get_events()
        ]

