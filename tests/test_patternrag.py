"""Tests for PatternRAG Strategy, Decorator, Observer, and Factory patterns."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from shared.types import Document, ScoredDoc
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from patternrag.strategy.base import RetrieverStrategy
from patternrag.strategy.bm25_retriever import BM25Retriever
from patternrag.strategy.dense_retriever import DenseRetriever
from patternrag.strategy.hybrid_retriever import HybridRetriever
from patternrag.decorator.base import RetrieverDecorator
from patternrag.decorator.caching_decorator import CachingDecorator
from patternrag.decorator.logging_decorator import LoggingDecorator
from patternrag.observer.base import PipelineEvent, PipelineObserver, EventDispatcher
from patternrag.observer.latency_logger import LatencyLogger
from patternrag.observer.metrics_collector import MetricsCollector
from patternrag.observer.cost_monitor import CostMonitor


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_corpus():
    return [
        Document(id="doc_0", text="The cat sat on the mat"),
        Document(id="doc_1", text="Dogs are friendly animals"),
        Document(id="doc_2", text="Python is a programming language"),
        Document(id="doc_3", text="Machine learning needs data"),
        Document(id="doc_4", text="The cat chased the dog"),
    ]


@pytest.fixture
def bm25_idx(small_corpus):
    idx = BM25Index()
    idx.build(small_corpus)
    return idx


class FakeEmbedder:
    def encode(self, texts):
        import hashlib
        items = [texts] if isinstance(texts, str) else texts
        vecs = []
        for t in items:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(384).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def faiss_idx(small_corpus):
    embedder = FakeEmbedder()
    idx = FAISSIndex()
    idx.build(small_corpus, embedder)
    return idx


@pytest.fixture
def embedder():
    return FakeEmbedder()


# ── Strategy tests ─────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_is_retriever_strategy(self, bm25_idx):
        r = BM25Retriever(bm25_idx)
        assert isinstance(r, RetrieverStrategy)

    def test_retrieve_returns_scored_docs(self, bm25_idx):
        r = BM25Retriever(bm25_idx)
        results = r.retrieve("cat mat", k=3)
        assert len(results) == 3
        assert all(isinstance(sd, ScoredDoc) for sd in results)

    def test_retrieve_top_result(self, bm25_idx):
        r = BM25Retriever(bm25_idx)
        results = r.retrieve("cat mat", k=5)
        # doc_0 or doc_4 has "cat"
        assert results[0].doc.id in {"doc_0", "doc_4"}


class TestDenseRetriever:
    def test_is_retriever_strategy(self, faiss_idx, embedder):
        r = DenseRetriever(faiss_idx, embedder)
        assert isinstance(r, RetrieverStrategy)

    def test_retrieve_returns_k_docs(self, faiss_idx, embedder):
        r = DenseRetriever(faiss_idx, embedder)
        for k in [1, 3, 5]:
            assert len(r.retrieve("cat", k=k)) == k

    def test_retrieve_scores_in_range(self, faiss_idx, embedder):
        r = DenseRetriever(faiss_idx, embedder)
        results = r.retrieve("python data", k=5)
        for sd in results:
            assert -1.01 <= sd.score <= 1.01


class TestHybridRetriever:
    def test_is_retriever_strategy(self, bm25_idx, faiss_idx, embedder):
        r = HybridRetriever(BM25Retriever(bm25_idx), DenseRetriever(faiss_idx, embedder))
        assert isinstance(r, RetrieverStrategy)

    def test_retrieve_returns_k_docs(self, bm25_idx, faiss_idx, embedder):
        r = HybridRetriever(BM25Retriever(bm25_idx), DenseRetriever(faiss_idx, embedder))
        results = r.retrieve("cat", k=4)
        assert len(results) == 4

    def test_rrf_scores_positive(self, bm25_idx, faiss_idx, embedder):
        r = HybridRetriever(BM25Retriever(bm25_idx), DenseRetriever(faiss_idx, embedder))
        results = r.retrieve("cat mat", k=5)
        for sd in results:
            assert sd.score > 0  # RRF scores are always positive

    def test_rrf_sorted_descending(self, bm25_idx, faiss_idx, embedder):
        r = HybridRetriever(BM25Retriever(bm25_idx), DenseRetriever(faiss_idx, embedder))
        results = r.retrieve("cat mat", k=5)
        scores = [sd.score for sd in results]
        assert scores == sorted(scores, reverse=True)


# ── Decorator tests ────────────────────────────────────────────────────────────

class FakeRetriever(RetrieverStrategy):
    def __init__(self, docs):
        self._docs = docs
        self.call_count = 0

    def retrieve(self, query: str, k: int):
        self.call_count += 1
        return self._docs[:k]


class TestRetrieverDecorator:
    def test_delegates_by_default(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        dec = LoggingDecorator(fake)
        results = dec.retrieve("test", k=3)
        assert len(results) == 3
        assert fake.call_count == 1

    def test_is_retriever_strategy(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        dec = LoggingDecorator(fake)
        assert isinstance(dec, RetrieverStrategy)


class TestCachingDecorator:
    def test_cache_miss_delegates(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        cached = CachingDecorator(fake)
        cached.retrieve("query", k=3)
        assert fake.call_count == 1

    def test_cache_hit_does_not_delegate(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        cached = CachingDecorator(fake)
        cached.retrieve("query", k=3)
        cached.retrieve("query", k=3)  # second call: cache hit
        assert fake.call_count == 1  # wrapped called only once

    def test_different_queries_not_cached(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        cached = CachingDecorator(fake)
        cached.retrieve("query1", k=3)
        cached.retrieve("query2", k=3)
        assert fake.call_count == 2

    def test_cache_hit_counter(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        cached = CachingDecorator(fake)
        cached.retrieve("q", k=2)
        cached.retrieve("q", k=2)
        assert cached.cache_hits == 1
        assert cached.cache_misses == 1

    def test_clear_cache(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        cached = CachingDecorator(fake)
        cached.retrieve("q", k=2)
        cached.clear_cache()
        cached.retrieve("q", k=2)  # should miss after clear
        assert fake.call_count == 2

    def test_composition_caching_wraps_logging(self, small_corpus):
        fake = FakeRetriever([ScoredDoc(doc=d, score=1.0) for d in small_corpus])
        composed = CachingDecorator(LoggingDecorator(fake))
        composed.retrieve("q", k=2)
        composed.retrieve("q", k=2)  # cache hit — inner logging not called again
        assert fake.call_count == 1


# ── Observer tests ─────────────────────────────────────────────────────────────

class TestPipelineEvent:
    def test_immutable(self):
        event = PipelineEvent(event_type="TEST", payload={"x": 1})
        with pytest.raises((AttributeError, TypeError)):
            event.event_type = "OTHER"  # type: ignore

    def test_timestamp_set(self):
        import time
        before = time.time()
        event = PipelineEvent(event_type="TEST")
        after = time.time()
        assert before <= event.timestamp <= after


class TestEventDispatcher:
    def test_register_and_notify(self):
        received = []

        class TestObserver(PipelineObserver):
            def on_event(self, event):
                received.append(event.event_type)

        dispatcher = EventDispatcher()
        dispatcher.register(TestObserver())
        dispatcher.notify(PipelineEvent("HELLO"))
        assert received == ["HELLO"]

    def test_multiple_observers(self):
        counts = [0, 0]

        class Obs1(PipelineObserver):
            def on_event(self, event): counts[0] += 1

        class Obs2(PipelineObserver):
            def on_event(self, event): counts[1] += 1

        d = EventDispatcher()
        d.register(Obs1())
        d.register(Obs2())
        d.notify(PipelineEvent("X"))
        assert counts == [1, 1]

    def test_faulty_observer_does_not_break_others(self):
        results = []

        class FaultyObserver(PipelineObserver):
            def on_event(self, event): raise RuntimeError("intentional")

        class GoodObserver(PipelineObserver):
            def on_event(self, event): results.append(event.event_type)

        d = EventDispatcher()
        d.register(FaultyObserver())
        d.register(GoodObserver())
        d.notify(PipelineEvent("TEST"))  # should not raise
        assert results == ["TEST"]

    def test_observer_count(self):
        d = EventDispatcher()
        assert d.observer_count == 0
        d.register(LatencyLogger())
        assert d.observer_count == 1


class TestLatencyLogger:
    def test_records_retrieval_latency(self):
        logger = LatencyLogger()
        logger.on_event(PipelineEvent("RETRIEVAL_DONE", {"elapsed_ms": 42.0}))
        assert logger.retrieval_latencies == [42.0]

    def test_records_generation_latency(self):
        logger = LatencyLogger()
        logger.on_event(PipelineEvent("GENERATION_DONE", {"elapsed_ms": 300.0}))
        assert logger.generation_latencies == [300.0]

    def test_ignores_other_events(self):
        logger = LatencyLogger()
        logger.on_event(PipelineEvent("QUERY_RECEIVED", {}))
        assert logger.retrieval_latencies == []

    def test_summary_statistics(self):
        logger = LatencyLogger()
        for ms in [10.0, 20.0, 30.0]:
            logger.on_event(PipelineEvent("RETRIEVAL_DONE", {"elapsed_ms": ms}))
        summary = logger.get_summary()
        assert summary["retrieval_latency_ms"]["mean"] == 20.0


class TestMetricsCollector:
    def test_records_retrieval_done(self):
        mc = MetricsCollector()
        mc.on_event(PipelineEvent("RETRIEVAL_DONE", {"query": "q", "retrieved_ids": ["d1", "d2"]}))
        assert len(mc.records) == 1
        assert mc.records[0]["n_docs"] == 2

    def test_ignores_other_events(self):
        mc = MetricsCollector()
        mc.on_event(PipelineEvent("GENERATION_DONE", {}))
        assert len(mc.records) == 0


class TestCostMonitor:
    def test_accumulates_tokens(self):
        cm = CostMonitor()
        cm.on_event(PipelineEvent("GENERATION_DONE", {"prompt_tokens": 100, "completion_tokens": 50}))
        cm.on_event(PipelineEvent("GENERATION_DONE", {"prompt_tokens": 200, "completion_tokens": 80}))
        summary = cm.get_summary()
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 130
        assert summary["n_generations"] == 2
        assert summary["estimated_cost_usd"] > 0


# ── Factory tests ──────────────────────────────────────────────────────────────

class TestPipelineFactory:
    def _make_configs(self, retriever_type="hybrid"):
        exp_cfg = {
            "retrieval": {"top_k": 5, "rrf_k": 60},
            "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "llm": {"provider": "groq", "model_name": "llama-3.3-70b-versatile", "max_tokens": 50, "temperature": 0.0},
        }
        pat_cfg = {
            "retriever": {"type": retriever_type},
            "decorators": [{"type": "logging"}, {"type": "caching"}],
            "observers": [{"type": "latency_logger"}, {"type": "metrics_collector"}],
        }
        return exp_cfg, pat_cfg

    def test_factory_builds_pipeline(self, bm25_idx, faiss_idx):
        from patternrag.factory.pipeline_factory import PipelineFactory
        from patternrag.pipeline import PatternRAGPipeline

        exp_cfg, pat_cfg = self._make_configs()
        factory = PipelineFactory()

        with patch("patternrag.factory.pipeline_factory.LLMGenerator") as MockLLM:
            MockLLM.return_value = MagicMock()
            pipeline = factory.build(
                exp_cfg=exp_cfg,
                pat_cfg=pat_cfg,
                bm25_index=bm25_idx,
                faiss_index=faiss_idx,
                embedder=FakeEmbedder(),
            )
        assert isinstance(pipeline, PatternRAGPipeline)

    def test_factory_registers_observers(self, bm25_idx, faiss_idx):
        from patternrag.factory.pipeline_factory import PipelineFactory

        exp_cfg, pat_cfg = self._make_configs()
        factory = PipelineFactory()

        with patch("patternrag.factory.pipeline_factory.LLMGenerator") as MockLLM:
            MockLLM.return_value = MagicMock()
            pipeline = factory.build(
                exp_cfg=exp_cfg,
                pat_cfg=pat_cfg,
                bm25_index=bm25_idx,
                faiss_index=faiss_idx,
                embedder=FakeEmbedder(),
            )
        assert pipeline._dispatcher.observer_count == 2  # latency_logger + metrics_collector

    def test_factory_unknown_retriever_raises(self, bm25_idx, faiss_idx):
        from patternrag.factory.pipeline_factory import PipelineFactory

        exp_cfg, pat_cfg = self._make_configs()
        pat_cfg["retriever"]["type"] = "unknown"
        factory = PipelineFactory()

        with patch("patternrag.factory.pipeline_factory.LLMGenerator"):
            with pytest.raises(ValueError, match="Unknown retriever type"):
                factory.build(
                    exp_cfg=exp_cfg,
                    pat_cfg=pat_cfg,
                    bm25_index=bm25_idx,
                    faiss_index=faiss_idx,
                    embedder=FakeEmbedder(),
                )

    def test_factory_bm25_only(self, bm25_idx, faiss_idx):
        from patternrag.factory.pipeline_factory import PipelineFactory

        exp_cfg, pat_cfg = self._make_configs(retriever_type="bm25")
        pat_cfg["decorators"] = []
        pat_cfg["observers"] = []
        factory = PipelineFactory()

        with patch("patternrag.factory.pipeline_factory.LLMGenerator") as MockLLM:
            MockLLM.return_value = MagicMock()
            pipeline = factory.build(
                exp_cfg=exp_cfg,
                pat_cfg=pat_cfg,
                bm25_index=bm25_idx,
                faiss_index=faiss_idx,
                embedder=FakeEmbedder(),
            )
        assert pipeline._dispatcher.observer_count == 0
