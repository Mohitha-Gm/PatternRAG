"""
Unit tests for Experiment M1 (TF-IDF Retrieval Strategy).

Tests:
1. TFIDFIndex build, search, save, and load.
2. TFIDFRetriever implements RetrieverStrategy and searches correctly.
3. PipelineFactory builds a PatternRAG pipeline with TFIDFRetriever when configured.
4. MonolithicRAGPipeline supports tfidf mode and returns correct results.
5. Equivalence: Monolithic and PatternRAG retrieve the identical documents under TF-IDF.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.types import Document, ScoredDoc
from shared.indexing.tfidf_index import TFIDFIndex
from patternrag.strategy.tfidf_retriever import TFIDFRetriever
from patternrag.factory.pipeline_factory import PipelineFactory
from patternrag.pipeline import PatternRAGPipeline
from monolithic.pipeline import MonolithicRAGPipeline


@pytest.fixture
def sample_corpus() -> list[Document]:
    return [
        Document(id="doc1", text="Quantum computing uses qubits to perform complex calculations exponentially faster.", metadata={"title": "Quantum Computing"}),
        Document(id="doc2", text="Machine learning algorithms build models based on sample training data to make predictions.", metadata={"title": "Machine Learning"}),
        Document(id="doc3", text="Deep neural networks learn representations of data with multiple abstraction layers.", metadata={"title": "Deep Learning"}),
        Document(id="doc4", text="Natural language processing involves computational linguistics and machine learning for text understanding.", metadata={"title": "Natural Language Processing"}),
    ]


@pytest.fixture
def tfidf_index(sample_corpus) -> TFIDFIndex:
    idx = TFIDFIndex()
    idx.build(sample_corpus)
    return idx


def test_tfidf_index_search(tfidf_index):
    results = tfidf_index.search("quantum calculations", k=2)
    assert len(results) == 2
    assert results[0].doc.id == "doc1"
    assert results[0].score > results[1].score


def test_tfidf_index_serialization(tfidf_index, sample_corpus):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "tfidf.pkl"
        tfidf_index.save(save_path)
        assert save_path.exists()

        loaded_index = TFIDFIndex.load(save_path)
        assert loaded_index.size == len(sample_corpus)
        res_orig = tfidf_index.search("machine learning", k=2)
        res_loaded = loaded_index.search("machine learning", k=2)
        assert [r.doc.id for r in res_orig] == [r.doc.id for r in res_loaded]
        assert [r.score for r in res_orig] == [r.score for r in res_loaded]


def test_patternrag_tfidf_retriever(tfidf_index):
    retriever = TFIDFRetriever(tfidf_index)
    results = retriever.retrieve("neural networks and representations", k=2)
    assert len(results) == 2
    assert results[0].doc.id == "doc3"


def test_patternrag_factory_with_tfidf(tfidf_index, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    factory = PipelineFactory()
    exp_cfg = {
        "retrieval": {"top_k": 2},
        "llm": {"model_name": "mock", "max_tokens": 10},
    }
    pat_cfg = {
        "retriever": {"type": "tfidf"},
        "decorators": [],
        "observers": [],
    }
    pipeline = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg,
        bm25_index=MagicMock(),
        faiss_index=MagicMock(),
        embedder=MagicMock(),
        tfidf_index=tfidf_index,
    )
    assert isinstance(pipeline, PatternRAGPipeline)
    assert isinstance(pipeline._retriever, TFIDFRetriever)


def test_monolithic_and_patternrag_tfidf_equivalence(tfidf_index, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    # Mock generator that returns a dummy answer
    generator = MagicMock()
    generator.generate.return_value = ("Test Answer", {"prompt_tokens": 10, "completion_tokens": 5})

    # Monolithic pipeline
    mono_config = {"retrieval_mode": "tfidf", "retrieval": {"top_k": 3}}
    mono_pipeline = MonolithicRAGPipeline(
        bm25_index=MagicMock(),
        faiss_index=MagicMock(),
        embedder=MagicMock(),
        generator=generator,
        config=mono_config,
        tfidf_index=tfidf_index,
    )

    # PatternRAG pipeline
    factory = PipelineFactory()
    exp_cfg = {"retrieval": {"top_k": 3}, "llm": {}}
    pat_cfg = {"retriever": {"type": "tfidf"}, "decorators": [], "observers": []}
    pat_pipeline = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg,
        bm25_index=MagicMock(),
        faiss_index=MagicMock(),
        embedder=MagicMock(),
        tfidf_index=tfidf_index,
    )
    pat_pipeline._generator = generator

    query = "computational linguistics and text"
    mono_res = mono_pipeline.run(query, k=3)
    pat_res = pat_pipeline.run(query, k=3)

    mono_ids = [sd.doc.id for sd in mono_res["retrieved_docs"]]
    pat_ids = [sd.doc.id for sd in pat_res["retrieved_docs"]]
    assert mono_ids == pat_ids, f"ID mismatch: {mono_ids} vs {pat_ids}"

    mono_scores = [round(sd.score, 6) for sd in mono_res["retrieved_docs"]]
    pat_scores = [round(sd.score, 6) for sd in pat_res["retrieved_docs"]]
    assert mono_scores == pat_scores, f"Score mismatch: {mono_scores} vs {pat_scores}"
