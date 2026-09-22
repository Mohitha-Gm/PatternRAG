"""
test_m2_filter.py — Validation and Equivalence Tests for M2 Result Filter.

Tests:
1. Documents above threshold are retained.
2. Documents below threshold are removed.
3. Top-1 fallback works when every score is below threshold.
4. PatternRAG ResultFilterDecorator works as an isolated decorator.
5. Monolithic filtering works in MonolithicRAGPipeline.
6. Both architectures produce identical filtered results for identical inputs.
7. Existing functionality remains unaffected when filtering is disabled.
8. PipelineFactory correctly instantiates and configures ResultFilterDecorator.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import pytest

from shared.types import Document, ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.decorator.result_filter_decorator import ResultFilterDecorator
from patternrag.factory.pipeline_factory import PipelineFactory
from monolithic.pipeline import MonolithicRAGPipeline


class DummyRetriever(RetrieverStrategy):
    """Stub retriever returning pre-configured ScoredDoc objects."""

    def __init__(self, docs: list[ScoredDoc]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        return self.docs[:k]


@pytest.fixture
def sample_scored_docs() -> list[ScoredDoc]:
    return [
        ScoredDoc(doc=Document(id="doc_1", text="Doc one content", metadata={}), score=0.85),
        ScoredDoc(doc=Document(id="doc_2", text="Doc two content", metadata={}), score=0.65),
        ScoredDoc(doc=Document(id="doc_3", text="Doc three content", metadata={}), score=0.45),
        ScoredDoc(doc=Document(id="doc_4", text="Doc four content", metadata={}), score=0.25),
    ]


def test_filter_retention_and_removal(sample_scored_docs: list[ScoredDoc]) -> None:
    """Requirement 1 & 2: Retain docs >= threshold, remove docs < threshold."""
    retriever = DummyRetriever(sample_scored_docs)
    filtered_retriever = ResultFilterDecorator(retriever, score_threshold=0.50)

    results = filtered_retriever.retrieve("test query", k=4)
    assert len(results) == 2
    assert [sd.doc.id for sd in results] == ["doc_1", "doc_2"]
    assert results[0].score == 0.85
    assert results[1].score == 0.65


def test_filter_top1_fallback(sample_scored_docs: list[ScoredDoc]) -> None:
    """Requirement 3: Top-1 fallback when all documents fall below threshold."""
    # All scores (0.85, 0.65, 0.45, 0.25) are below 0.95
    retriever = DummyRetriever(sample_scored_docs)
    filtered_retriever = ResultFilterDecorator(retriever, score_threshold=0.95)

    results = filtered_retriever.retrieve("test query", k=4)
    assert len(results) == 1
    assert results[0].doc.id == "doc_1"
    assert results[0].score == 0.85


def test_filter_empty_list() -> None:
    """Filtering on an empty list returns an empty list without error."""
    retriever = DummyRetriever([])
    filtered_retriever = ResultFilterDecorator(retriever, score_threshold=0.50)

    results = filtered_retriever.retrieve("test query", k=4)
    assert results == []


def test_patternrag_result_filter_decorator_properties(sample_scored_docs: list[ScoredDoc]) -> None:
    """Requirement 4: ResultFilterDecorator conforms to interface and properties."""
    retriever = DummyRetriever(sample_scored_docs)
    dec = ResultFilterDecorator(retriever, score_threshold=0.60)
    assert dec.score_threshold == 0.60

    res = dec.retrieve("query", k=10)
    assert len(res) == 2
    assert [sd.doc.id for sd in res] == ["doc_1", "doc_2"]


def test_monolithic_filter_results_helper(sample_scored_docs: list[ScoredDoc]) -> None:
    """Requirement 5: Monolithic static helper functions identically."""
    # Above threshold
    res1 = MonolithicRAGPipeline._filter_results(sample_scored_docs, score_threshold=0.50)
    assert len(res1) == 2
    assert [sd.doc.id for sd in res1] == ["doc_1", "doc_2"]

    # All below threshold fallback
    res2 = MonolithicRAGPipeline._filter_results(sample_scored_docs, score_threshold=0.95)
    assert len(res2) == 1
    assert res2[0].doc.id == "doc_1"

    # Empty list
    res3 = MonolithicRAGPipeline._filter_results([], score_threshold=0.50)
    assert res3 == []


def test_monolithic_and_patternrag_filter_equivalence(
    sample_scored_docs: list[ScoredDoc],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 6: Monolithic and PatternRAG produce identical filtered results."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_api_key_for_test")

    # Threshold 0.50: partial filtering
    mono_filtered = MonolithicRAGPipeline._filter_results(sample_scored_docs, score_threshold=0.50)
    pattern_dec = ResultFilterDecorator(DummyRetriever(sample_scored_docs), score_threshold=0.50)
    pattern_filtered = pattern_dec.retrieve("query", k=10)

    assert len(mono_filtered) == len(pattern_filtered)
    assert [sd.doc.id for sd in mono_filtered] == [sd.doc.id for sd in pattern_filtered]
    assert [sd.score for sd in mono_filtered] == [sd.score for sd in pattern_filtered]

    # Threshold 0.99: top-1 fallback
    mono_fallback = MonolithicRAGPipeline._filter_results(sample_scored_docs, score_threshold=0.99)
    pattern_fallback_dec = ResultFilterDecorator(DummyRetriever(sample_scored_docs), score_threshold=0.99)
    pattern_fallback = pattern_fallback_dec.retrieve("query", k=10)

    assert len(mono_fallback) == 1
    assert len(pattern_fallback) == 1
    assert mono_fallback[0].doc.id == pattern_fallback[0].doc.id == "doc_1"
    assert mono_fallback[0].score == pattern_fallback[0].score == 0.85


def test_filter_disabled_backward_compatibility(sample_scored_docs: list[ScoredDoc]) -> None:
    """Requirement 7: Existing functionality unaffected when filtering is disabled."""
    # In monolithic with score_threshold=None, no filter applied
    bm25_index = MagicMock()
    bm25_index.search.return_value = sample_scored_docs
    pipeline = MonolithicRAGPipeline(
        bm25_index=bm25_index,
        faiss_index=MagicMock(),
        embedder=MagicMock(),
        generator=MagicMock(),
        config={"retrieval_mode": "bm25"},
        score_threshold=None,
    )
    assert pipeline._score_threshold is None


def test_pipeline_factory_registers_and_builds_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 8: PipelineFactory wires ResultFilterDecorator via pat_cfg."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")

    factory = PipelineFactory()
    assert "result_filter" in factory._DECORATOR_TYPES
    assert "filter" in factory._DECORATOR_TYPES

    pat_cfg = {
        "retriever": {"type": "bm25"},
        "decorators": [
            {"type": "result_filter", "score_threshold": 0.75}
        ],
        "observers": []
    }
    exp_cfg = {
        "retrieval": {"top_k": 5, "rrf_k": 60},
        "llm": {"model": "llama-3.3-70b-versatile", "temperature": 0.0, "max_tokens": 100},
    }

    bm25_index = MagicMock()
    bm25_index.search.return_value = []
    faiss_index = MagicMock()
    embedder = MagicMock()

    pipeline = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg,
        bm25_index=bm25_index,
        faiss_index=faiss_index,
        embedder=embedder,
    )

    # Check that decorator wrapped the retriever
    retriever = pipeline._retriever
    assert isinstance(retriever, ResultFilterDecorator)
    assert retriever.score_threshold == 0.75
