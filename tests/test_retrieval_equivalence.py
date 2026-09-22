"""
Retrieval Equivalence Test.

CRITICAL TEST: Verifies that for the same query, top_k, and indexes,
the Monolithic pipeline's hybrid RRF retrieval and PatternRAG's
HybridRetriever return IDENTICAL document IDs in the SAME order.

This test proves that the architectural restructuring did not change the
retrieval algorithm, which is required for a fair architectural comparison.

Note: LLM outputs are NOT compared here — they are inherently non-
deterministic and are not expected to be identical across separate calls.
"""
import pytest
import numpy as np

from shared.types import Document, ScoredDoc
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from patternrag.strategy.bm25_retriever import BM25Retriever
from patternrag.strategy.dense_retriever import DenseRetriever
from patternrag.strategy.hybrid_retriever import HybridRetriever
from monolithic.pipeline import MonolithicRAGPipeline


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


class FakeGenerator:
    """Stub generator — not used in retrieval equivalence tests."""
    def generate(self, prompt: str):
        return "FAKE ANSWER", {"prompt_tokens": 0, "completion_tokens": 0}


@pytest.fixture(scope="module")
def shared_corpus():
    return [
        Document(id=f"doc_{i}", text=t)
        for i, t in enumerate([
            "The cat sat on the mat and looked around carefully",
            "Dogs are friendly animals that love to play fetch",
            "Python is a popular programming language for data science",
            "Machine learning models require large amounts of training data",
            "The cat chased the dog down the street very quickly",
            "Natural language processing uses text as input",
            "Deep learning requires GPU hardware for training efficiency",
            "The quick brown fox jumped over the lazy dog yesterday",
        ])
    ]


@pytest.fixture(scope="module")
def shared_bm25(shared_corpus):
    idx = BM25Index()
    idx.build(shared_corpus)
    return idx


@pytest.fixture(scope="module")
def shared_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="module")
def shared_faiss(shared_corpus, shared_embedder):
    idx = FAISSIndex()
    idx.build(shared_corpus, shared_embedder)
    return idx


@pytest.fixture(scope="module")
def monolithic_pipeline(shared_bm25, shared_faiss, shared_embedder):
    config = {"retrieval": {"top_k": 5, "rrf_k": 60}, "retrieval_mode": "hybrid"}
    return MonolithicRAGPipeline(
        bm25_index=shared_bm25,
        faiss_index=shared_faiss,
        embedder=shared_embedder,
        generator=FakeGenerator(),
        config=config,
    )


@pytest.fixture(scope="module")
def hybrid_retriever(shared_bm25, shared_faiss, shared_embedder):
    bm25_r = BM25Retriever(shared_bm25)
    dense_r = DenseRetriever(shared_faiss, shared_embedder)
    return HybridRetriever(bm25_r, dense_r, rrf_k=60)


TEST_QUERIES = [
    "cat on the mat",
    "python programming language",
    "machine learning training data",
    "dog chased by cat",
    "natural language processing",
    "deep learning GPU",
]


class TestRetrievalEquivalence:
    """
    Assert that Monolithic hybrid retrieval == PatternRAG HybridRetriever.

    For each test query, both systems must return exactly the same
    document IDs in exactly the same order.
    """

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_document_ids_identical(
        self, query, monolithic_pipeline, hybrid_retriever
    ):
        k = 5
        rrf_k = 60

        # Monolithic: call the internal _rrf_merge directly with same inputs
        bm25_results = monolithic_pipeline._bm25_index.search(query, k)
        query_vec = monolithic_pipeline._embedder.encode(query)
        dense_results = monolithic_pipeline._faiss_index.search(query_vec, k)
        mono_docs = monolithic_pipeline._rrf_merge(bm25_results, dense_results, k)
        mono_ids = [sd.doc.id for sd in mono_docs]

        # PatternRAG: HybridRetriever
        prag_docs = hybrid_retriever.retrieve(query, k)
        prag_ids = [sd.doc.id for sd in prag_docs]

        assert mono_ids == prag_ids, (
            f"Retrieval mismatch for query: {query!r}\n"
            f"  Monolithic: {mono_ids}\n"
            f"  PatternRAG: {prag_ids}"
        )

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_rrf_scores_identical(
        self, query, monolithic_pipeline, hybrid_retriever
    ):
        """Verify RRF scores are numerically identical (not just rank-order)."""
        k = 5

        bm25_results = monolithic_pipeline._bm25_index.search(query, k)
        query_vec = monolithic_pipeline._embedder.encode(query)
        dense_results = monolithic_pipeline._faiss_index.search(query_vec, k)
        mono_docs = monolithic_pipeline._rrf_merge(bm25_results, dense_results, k)
        mono_scores = {sd.doc.id: sd.score for sd in mono_docs}

        prag_docs = hybrid_retriever.retrieve(query, k)
        prag_scores = {sd.doc.id: sd.score for sd in prag_docs}

        for doc_id, mono_score in mono_scores.items():
            prag_score = prag_scores.get(doc_id, None)
            assert prag_score is not None, f"Doc {doc_id!r} missing from PatternRAG results"
            assert abs(mono_score - prag_score) < 1e-9, (
                f"Score mismatch for doc {doc_id!r}: mono={mono_score}, prag={prag_score}"
            )
