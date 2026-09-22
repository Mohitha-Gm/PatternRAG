"""Tests for shared infrastructure (BM25, FAISS, metrics) using small fixtures."""
import pytest
import numpy as np
from shared.types import Document, ScoredDoc
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from shared.evaluation.retrieval_metrics import compute_retrieval_metrics, aggregate_retrieval_metrics
from shared.evaluation.latency_tracker import LatencyTracker
from shared.prompts.templates import build_prompt


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_corpus():
    return [
        Document(id="doc_0", text="The cat sat on the mat and looked around"),
        Document(id="doc_1", text="Dogs are friendly animals that love to play fetch"),
        Document(id="doc_2", text="Python is a popular programming language for data science"),
        Document(id="doc_3", text="Machine learning models require large amounts of training data"),
        Document(id="doc_4", text="The cat chased the dog down the street quickly"),
    ]


@pytest.fixture
def bm25_index(small_corpus):
    idx = BM25Index()
    idx.build(small_corpus)
    return idx


class FakeEmbedder:
    """Deterministic embedder for testing (random but seeded per text)."""
    def encode(self, texts):
        import hashlib
        vecs = []
        for t in [texts] if isinstance(texts, str) else texts:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(384).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def faiss_index(small_corpus):
    embedder = FakeEmbedder()
    idx = FAISSIndex()
    idx.build(small_corpus, embedder)
    return idx


# ── BM25 tests ─────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_sets_size(self, bm25_index, small_corpus):
        assert bm25_index.size == len(small_corpus)

    def test_search_returns_scored_docs(self, bm25_index):
        results = bm25_index.search("cat mat", k=3)
        assert len(results) == 3
        assert all(isinstance(r, ScoredDoc) for r in results)

    def test_search_top_result_relevant(self, bm25_index):
        results = bm25_index.search("cat mat", k=5)
        top_ids = [r.doc.id for r in results]
        # "cat" appears in doc_0 and doc_4; one of them should be first
        assert top_ids[0] in {"doc_0", "doc_4"}

    def test_search_respects_k(self, bm25_index):
        for k in [1, 3, 5]:
            assert len(bm25_index.search("cat", k=k)) == k

    def test_save_and_load(self, bm25_index, tmp_path, small_corpus):
        path = tmp_path / "bm25.pkl"
        bm25_index.save(path)
        loaded = BM25Index.load(path)
        assert loaded.size == len(small_corpus)
        r1 = bm25_index.search("cat", k=3)
        r2 = loaded.search("cat", k=3)
        assert [r.doc.id for r in r1] == [r.doc.id for r in r2]

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BM25Index.load(tmp_path / "nonexistent.pkl")


# ── FAISS tests ────────────────────────────────────────────────────────────────

class TestFAISSIndex:
    def test_build_sets_size(self, faiss_index, small_corpus):
        assert faiss_index.size == len(small_corpus)

    def test_search_returns_scored_docs(self, faiss_index):
        q = FakeEmbedder().encode("cat mat")
        results = faiss_index.search(q, k=3)
        assert len(results) == 3
        assert all(isinstance(r, ScoredDoc) for r in results)

    def test_search_scores_bounded(self, faiss_index):
        q = FakeEmbedder().encode("python programming")
        results = faiss_index.search(q, k=5)
        for r in results:
            assert -1.01 <= r.score <= 1.01  # cosine similarity range

    def test_save_and_load(self, faiss_index, tmp_path, small_corpus):
        path = tmp_path / "faiss.bin"
        faiss_index.save(path)
        loaded = FAISSIndex.load(path)
        assert loaded.size == len(small_corpus)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FAISSIndex.load(tmp_path / "missing.bin")


# ── Metrics tests ──────────────────────────────────────────────────────────────

class TestRetrievalMetrics:
    def test_perfect_recall(self):
        m = compute_retrieval_metrics(["a", "b", "c"], ["a", "b"], k_values=[2])
        assert m["Recall@2"] == 1.0

    def test_zero_recall(self):
        m = compute_retrieval_metrics(["x", "y", "z"], ["a", "b"], k_values=[3])
        assert m["Recall@3"] == 0.0

    def test_mrr_first(self):
        m = compute_retrieval_metrics(["gold", "other"], ["gold"])
        assert m["MRR"] == 1.0

    def test_mrr_second(self):
        m = compute_retrieval_metrics(["other", "gold"], ["gold"])
        assert abs(m["MRR"] - 0.5) < 1e-9

    def test_ndcg(self):
        m = compute_retrieval_metrics(["a", "b", "c", "d"], ["a", "c"])
        assert 0.0 <= m["NDCG@10"] <= 1.0

    def test_empty_gold(self):
        m = compute_retrieval_metrics(["a", "b"], [])
        assert m["Recall@1"] == 0.0

    def test_aggregate(self):
        records = [
            {"Recall@1": 0.5, "MRR": 0.5},
            {"Recall@1": 1.0, "MRR": 1.0},
        ]
        agg = aggregate_retrieval_metrics(records)
        assert abs(agg["Recall@1"] - 0.75) < 1e-9


# ── LatencyTracker tests ───────────────────────────────────────────────────────

class TestLatencyTracker:
    def test_records_elapsed(self):
        import time
        with LatencyTracker() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10.0

    def test_elapsed_zero_before_exit(self):
        tracker = LatencyTracker()
        assert tracker.elapsed_ms == 0.0


# ── Prompt template tests ──────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_question(self):
        docs = [ScoredDoc(doc=Document(id="d", text="some text"), score=1.0)]
        prompt = build_prompt("What is the capital?", docs)
        assert "What is the capital?" in prompt

    def test_contains_context(self):
        docs = [ScoredDoc(doc=Document(id="d", text="Paris is the capital"), score=1.0)]
        prompt = build_prompt("Capital?", docs)
        assert "Paris is the capital" in prompt

    def test_multiple_docs_numbered(self):
        docs = [
            ScoredDoc(doc=Document(id="d1", text="Doc one content"), score=1.0),
            ScoredDoc(doc=Document(id="d2", text="Doc two content"), score=0.5),
        ]
        prompt = build_prompt("Q?", docs)
        assert "[1]" in prompt
        assert "[2]" in prompt
