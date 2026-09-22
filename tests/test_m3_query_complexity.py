"""
test_m3_query_complexity.py — Validation and Equivalence Tests for M3 Query Complexity Monitor.

Tests:
1. Normal query length is recorded.
2. Short query (<15) is counted as an outlier.
3. Long query (>200) is counted as an outlier.
4. Multiple queries produce correct mean/min/max.
5. Observer responds correctly to QUERY_RECEIVED (and ignores other events).
6. Observer does not interfere with the query pipeline (transparent, non-intrusive).
7. Monolithic and PatternRAG produce equivalent monitoring statistics for identical query sequences.
8. Existing behavior remains unaffected when the observer is not enabled.
9. PipelineFactory properly registers and instantiates QueryComplexityMonitor.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import pytest

from patternrag.observer.base import EventDispatcher, PipelineEvent
from patternrag.observer.query_complexity_monitor import QueryComplexityMonitor
from patternrag.factory.pipeline_factory import PipelineFactory
from monolithic.pipeline import MonolithicRAGPipeline
from shared.types import Document, ScoredDoc


def test_normal_query_length_recorded() -> None:
    """Requirement 1: Normal query length is recorded without outlier flag."""
    monitor = QueryComplexityMonitor()
    query = "What is the capital of France and what is its history?"  # 54 chars
    event = PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": query, "k": 5})

    monitor.on_event(event)

    summary = monitor.get_summary()
    assert summary["mean_query_length"] == 54.0
    assert summary["min_query_length"] == 54
    assert summary["max_query_length"] == 54
    assert summary["outlier_query_count"] == 0
    assert monitor.query_lengths == [54]


def test_short_query_outlier() -> None:
    """Requirement 2: Short query (<15) is counted as an outlier."""
    monitor = QueryComplexityMonitor()

    # 14 chars -> outlier
    q_short = "Short query???"
    assert len(q_short) == 14
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q_short, "k": 5}))

    # 15 chars -> boundary: not an outlier
    q_boundary = "123456789012345"
    assert len(q_boundary) == 15
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q_boundary, "k": 5}))

    summary = monitor.get_summary()
    assert summary["outlier_query_count"] == 1
    assert summary["min_query_length"] == 14
    assert summary["max_query_length"] == 15


def test_long_query_outlier() -> None:
    """Requirement 3: Long query (>200) is counted as an outlier."""
    monitor = QueryComplexityMonitor()

    # 200 chars -> boundary: not an outlier
    q_200 = "A" * 200
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q_200, "k": 5}))

    # 201 chars -> outlier
    q_201 = "B" * 201
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q_201, "k": 5}))

    summary = monitor.get_summary()
    assert summary["outlier_query_count"] == 1
    assert summary["min_query_length"] == 200
    assert summary["max_query_length"] == 201


def test_multiple_queries_mean_min_max() -> None:
    """Requirement 4: Multiple queries produce correct mean, min, max, and outlier stats."""
    monitor = QueryComplexityMonitor()

    # Empty state test
    empty_summary = monitor.get_summary()
    assert empty_summary["mean_query_length"] == 0.0
    assert empty_summary["min_query_length"] == 0
    assert empty_summary["max_query_length"] == 0
    assert empty_summary["outlier_query_count"] == 0

    queries = [
        "Tiny" * 2,          # 8 chars  (< 15 -> outlier)
        "Medium query" * 2,  # 24 chars (normal)
        "Longer question" * 4, # 60 chars (normal)
        "X" * 250,           # 250 chars (> 200 -> outlier)
    ]
    # Expected lengths: [8, 24, 60, 250] -> sum = 342 -> mean = 85.5
    for q in queries:
        monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q, "k": 5}))

    summary = monitor.get_summary()
    assert summary["mean_query_length"] == 85.5
    assert summary["min_query_length"] == 8
    assert summary["max_query_length"] == 250
    assert summary["outlier_query_count"] == 2


def test_observer_responds_only_to_query_received() -> None:
    """Requirement 5: Observer responds to QUERY_RECEIVED and ignores other events."""
    monitor = QueryComplexityMonitor()

    # Dispatched other event types
    monitor.on_event(PipelineEvent(event_type="RETRIEVAL_DONE", payload={"query": "X" * 300}))
    monitor.on_event(PipelineEvent(event_type="GENERATION_DONE", payload={"query": "Y" * 5}))
    monitor.on_event(PipelineEvent(event_type="CACHE_HIT", payload={"query": "Z" * 10}))

    # Should remain empty
    assert len(monitor.query_lengths) == 0
    assert monitor.get_summary()["outlier_query_count"] == 0

    # Dispatch QUERY_RECEIVED
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": "Valid query text"}))
    assert len(monitor.query_lengths) == 1


def test_observer_non_interference_and_exception_safety() -> None:
    """Requirement 6: Observer does not interfere with pipeline and handles anomalies gracefully."""
    monitor = QueryComplexityMonitor()

    # Anomaly payload (e.g. None or missing query)
    monitor.on_event(PipelineEvent(event_type="QUERY_RECEIVED", payload={}))
    # Should record len("") == 0 as an outlier (< 15) without raising exception
    assert monitor.outlier_query_count == 1
    assert monitor.query_lengths == [0]


def test_monolithic_and_patternrag_monitoring_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 7: Monolithic and PatternRAG produce identical monitoring statistics."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")

    test_queries = [
        "Short q?",                                           # 8 chars (<15, outlier)
        "What is the airspeed velocity of an unladen swallow?", # 52 chars (normal)
        "Explain the difference between Strategy and Decorator in GoF design patterns.", # 77 chars (normal)
        "Z" * 220,                                            # 220 chars (>200, outlier)
    ]

    # 1. Monolithic Pipeline
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        ScoredDoc(doc=Document(id="d1", text="Text 1", metadata={}), score=0.9)
    ]
    mock_generator = MagicMock()
    mock_generator.generate.return_value = ("Test answer", {"prompt_tokens": 10, "completion_tokens": 5})

    mono_pipeline = MonolithicRAGPipeline(
        bm25_index=mock_bm25,
        faiss_index=MagicMock(),
        embedder=MagicMock(),
        generator=mock_generator,
        config={"retrieval_mode": "bm25"},
    )

    for q in test_queries:
        mono_pipeline.run(q, k=1)

    mono_summary = mono_pipeline.get_query_complexity_summary()
    mono_full_summary = mono_pipeline.get_monitoring_summary()

    # 2. PatternRAG Observer
    monitor = QueryComplexityMonitor()
    dispatcher = EventDispatcher()
    dispatcher.register(monitor)

    for q in test_queries:
        dispatcher.notify(PipelineEvent(event_type="QUERY_RECEIVED", payload={"query": q, "k": 1}))

    pattern_summary = monitor.get_summary()

    # Verify equivalence
    assert mono_summary["mean_query_length"] == pattern_summary["mean_query_length"]
    assert mono_summary["min_query_length"] == pattern_summary["min_query_length"]
    assert mono_summary["max_query_length"] == pattern_summary["max_query_length"]
    assert mono_summary["outlier_query_count"] == pattern_summary["outlier_query_count"]

    assert mono_full_summary["mean_query_length"] == pattern_summary["mean_query_length"]
    assert mono_full_summary["min_query_length"] == pattern_summary["min_query_length"]
    assert mono_full_summary["max_query_length"] == pattern_summary["max_query_length"]
    assert mono_full_summary["outlier_query_count"] == pattern_summary["outlier_query_count"]


def test_behavior_unaffected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 8: PipelineFactory functions cleanly when observer is enabled or disabled."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    factory = PipelineFactory()

    assert "query_complexity_monitor" in factory._OBSERVER_TYPES
    assert "query_complexity" in factory._OBSERVER_TYPES

    # Configuration without QueryComplexityMonitor
    pat_cfg_without = {
        "retriever": {"type": "bm25"},
        "decorators": [],
        "observers": [{"type": "latency_logger"}]
    }
    exp_cfg = {
        "retrieval": {"top_k": 5, "rrf_k": 60},
        "llm": {"model": "llama-3.3-70b-versatile", "temperature": 0.0, "max_tokens": 100},
    }

    pipe_without = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg_without,
        bm25_index=MagicMock(),
        faiss_index=MagicMock(),
        embedder=MagicMock(),
    )
    assert pipe_without._dispatcher.observer_count == 1

    # Configuration with QueryComplexityMonitor
    pat_cfg_with = {
        "retriever": {"type": "bm25"},
        "decorators": [],
        "observers": [
            {"type": "latency_logger"},
            {"type": "query_complexity_monitor"}
        ]
    }
    pipe_with = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg_with,
        bm25_index=MagicMock(),
        faiss_index=MagicMock(),
        embedder=MagicMock(),
    )
    assert pipe_with._dispatcher.observer_count == 2
