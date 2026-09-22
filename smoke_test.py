"""Quick smoke test — verifies all imports and config correctness."""
import sys
sys.path.insert(0, ".")

# All imports
from shared.types import Document, ScoredDoc
from shared.config import load_yaml
from shared.data.hotpotqa_loader import load_hotpotqa
from shared.data.corpus_builder import build_corpus
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex
from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.llm.generator import LLMGenerator
from shared.prompts.templates import build_prompt
from shared.evaluation.retrieval_metrics import compute_retrieval_metrics
from shared.evaluation.ragas_evaluator import evaluate_with_ragas
from shared.evaluation.latency_tracker import LatencyTracker
from monolithic.pipeline import MonolithicRAGPipeline
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
from patternrag.factory.pipeline_factory import PipelineFactory
from patternrag.pipeline import PatternRAGPipeline
print("ALL IMPORTS: OK")

# Config correctness
cfg = load_yaml("configs/experiment.yaml")
pat = load_yaml("configs/patternrag.yaml")
assert "hybrid_alpha" not in str(cfg), "hybrid_alpha must be absent from config"
assert "rrf_k" in cfg["retrieval"], "rrf_k must be present"
assert cfg["retrieval"]["rrf_k"] == 60
assert pat["retriever"]["type"] == "hybrid"
assert len(pat["decorators"]) == 2
assert len(pat["observers"]) == 3
print(f"CONFIG: top_k={cfg['retrieval']['top_k']}, rrf_k={cfg['retrieval']['rrf_k']}, hybrid_alpha=ABSENT (correct)")
print(f"PATTERNRAG: retriever={pat['retriever']['type']}, decorators={len(pat['decorators'])}, observers={len(pat['observers'])}")

# No cross-imports between monolithic and patternrag
import ast, pathlib
mono_src = pathlib.Path("monolithic/pipeline.py").read_text()
prag_src  = pathlib.Path("patternrag/pipeline.py").read_text()
assert "from patternrag" not in mono_src and "import patternrag" not in mono_src
assert "from monolithic" not in prag_src and "import monolithic" not in prag_src
print("ISOLATION: monolithic/ and patternrag/ do not import from each other (correct)")

# Smoke test Strategy + Decorator + Observer + Factory wiring (no LLM needed)
import numpy as np
corpus = [Document(id=f"d{i}", text=f"document text number {i}") for i in range(10)]
bm25 = BM25Index().build(corpus)

class _Emb:
    def encode(self, texts):
        import hashlib
        items = [texts] if isinstance(texts, str) else texts
        vecs = []
        for t in items:
            s = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31)
            v = np.random.RandomState(s).randn(384).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(vecs, dtype=np.float32)

emb = _Emb()
faiss = FAISSIndex().build(corpus, emb)

# Strategy
bm25r = BM25Retriever(bm25)
denser = DenseRetriever(faiss, emb)
hybrid = HybridRetriever(bm25r, denser, rrf_k=60)
results = hybrid.retrieve("document text", k=5)
assert len(results) == 5 and all(r.score > 0 for r in results)
print("STRATEGY: BM25Retriever, DenseRetriever, HybridRetriever: OK")

# Decorator
cached = CachingDecorator(LoggingDecorator(hybrid))
r1 = cached.retrieve("document text", k=5)
r2 = cached.retrieve("document text", k=5)  # cache hit
assert cached.cache_hits == 1
print("DECORATOR: CachingDecorator(LoggingDecorator(HybridRetriever)): OK")

# Observer
dispatcher = EventDispatcher()
ll = LatencyLogger()
mc = MetricsCollector()
cm = CostMonitor()
dispatcher.register(ll)
dispatcher.register(mc)
dispatcher.register(cm)
dispatcher.notify(PipelineEvent("RETRIEVAL_DONE", {"elapsed_ms": 50.0, "retrieved_ids": ["d0","d1"], "query": "q"}))
dispatcher.notify(PipelineEvent("GENERATION_DONE", {"elapsed_ms": 300.0, "prompt_tokens": 100, "completion_tokens": 30}))
assert ll.retrieval_latencies == [50.0]
assert len(mc.records) == 1
assert cm.get_summary()["total_prompt_tokens"] == 100
print("OBSERVER: LatencyLogger, MetricsCollector, CostMonitor: OK")

# Factory (mock LLM)
from unittest.mock import patch, MagicMock
exp_cfg = {"retrieval": {"top_k": 5, "rrf_k": 60}, "llm": {"provider": "groq", "model_name": "llama-3.3-70b-versatile", "max_tokens": 50, "temperature": 0.0}}
pat_cfg = {"retriever": {"type": "hybrid"}, "decorators": [{"type": "logging"}, {"type": "caching"}], "observers": [{"type": "latency_logger"}]}
with patch("patternrag.factory.pipeline_factory.LLMGenerator") as MockLLM:
    MockLLM.return_value = MagicMock()
    pipeline = PipelineFactory().build(exp_cfg, pat_cfg, bm25, faiss, emb)
assert isinstance(pipeline, PatternRAGPipeline)
assert pipeline._dispatcher.observer_count == 1
print("FACTORY: PipelineFactory -> PatternRAGPipeline: OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
