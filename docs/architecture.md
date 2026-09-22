# PatternRAG — Architecture Documentation

## Research Context

PatternRAG compares two RAG implementations that use **identical underlying
algorithms and data** but differ only in software architecture:

| System | Architecture |
|---|---|
| Monolithic | All pipeline logic inline in one class |
| PatternRAG | Decomposed using four GoF design patterns |

The independent variable is software architecture.
The controlled variables are everything else (dataset, indexes, LLM, prompts, metrics).

---

## Shared Infrastructure (`shared/`)

Both systems import from `shared/` for all low-level operations.
`shared/` contains NO retrieval strategy logic and NO GoF patterns.

```
shared/
├── types.py          → Document, ScoredDoc dataclasses
├── config.py         → load_yaml(), get_api_key()
├── data/
│   ├── hotpotqa_loader.py   → load_hotpotqa(split, n_samples, seed)
│   └── corpus_builder.py    → build_corpus(split) → list[Document]
├── indexing/
│   ├── bm25_index.py        → BM25Index: build/save/load/search
│   └── faiss_index.py       → FAISSIndex: build/save/load/search
├── embedding/
│   └── embedder.py          → SentenceTransformerEmbedder
├── llm/
│   └── generator.py         → LLMGenerator (gpt-4o-mini)
├── prompts/
│   └── templates.py         → build_prompt(question, docs) → str
└── evaluation/
    ├── retrieval_metrics.py  → Recall@k, Precision@k, MRR, NDCG
    ├── ragas_evaluator.py    → RAGAS faithfulness/relevancy/precision/recall
    └── latency_tracker.py   → LatencyTracker context manager
```

---

## Pattern 1 — Strategy (Retrieval)

**Location**: `patternrag/strategy/`

**Intent**: Define a family of retrieval algorithms, encapsulate each one,
and make them interchangeable. The pipeline depends on an abstraction,
not a concrete retriever.

```
RetrieverStrategy (ABC)
    retrieve(query: str, k: int) → list[ScoredDoc]
         │
         ├── BM25Retriever      → calls BM25Index.search()
         ├── DenseRetriever     → encodes query → calls FAISSIndex.search()
         └── HybridRetriever    → composes BM25Retriever + DenseRetriever via RRF
```

**RRF formula** (standard, no alpha weight):
```
score(d) = Σ_i  1 / (rrf_k + rank_i(d))       rrf_k = 60
```

**Monolithic equivalent**: The same BM25 search, FAISS search, and RRF merge
are performed inline inside `MonolithicRAGPipeline.run()`.

**Extensibility**: To add a new retrieval strategy (M1 experiment), add one
file to `patternrag/strategy/` and update `patternrag.yaml`. No existing
file changes required.

---

## Pattern 2 — Decorator (Cross-cutting Concerns)

**Location**: `patternrag/decorator/`

**Intent**: Attach additional responsibilities (caching, logging) to
retrievers dynamically. Decorators are transparent wrappers that conform
to the same `RetrieverStrategy` interface.

```
RetrieverDecorator (ABC, extends RetrieverStrategy)
    _wrapped: RetrieverStrategy
    retrieve(query, k) → delegates to _wrapped
         │
         ├── CachingDecorator   → cache dict → _wrapped → cache store
         └── LoggingDecorator   → log start → _wrapped → log result
```

**Composition example** (as configured in `patternrag.yaml`):
```
CachingDecorator(
    LoggingDecorator(
        HybridRetriever(BM25Retriever, DenseRetriever)
    )
)
```

Each layer adds one concern; no existing class is modified.

**Monolithic equivalent**: Caching is a plain `dict` checked/stored inline.
Logging is `logger.debug(...)` statements inline in `run()`.

**Extensibility**: To add a new cross-cutting concern (M2 experiment), add
one file to `patternrag/decorator/` and register it in `patternrag.yaml`.

---

## Pattern 3 — Observer (Monitoring Side-Channel)

**Location**: `patternrag/observer/`

**Intent**: Define a one-to-many dependency between the pipeline (subject)
and monitoring components (observers). Observers react to pipeline events
without being part of the sequential data flow.

```
PatternRAGPipeline.run()
    │
    │  emits events at key stages
    ▼
EventDispatcher.notify(PipelineEvent)
    ├── LatencyLogger.on_event()     → records retrieval/generation ms
    ├── MetricsCollector.on_event()  → stores retrieved_ids per query
    └── CostMonitor.on_event()       → accumulates token counts + cost
```

**Event types**:
- `QUERY_RECEIVED` — fired when `run()` is called
- `RETRIEVAL_DONE` — fired after retrieval completes (includes `elapsed_ms`, `retrieved_ids`)
- `GENERATION_DONE` — fired after LLM returns (includes `elapsed_ms`, token counts)

**IMPORTANT**: Observers do NOT intercept or modify the RAG data flow.
They receive events asynchronously (logically; synchronously in practice).

**Monolithic equivalent**: Latency tracking and counters are inline
variables in `run()`. No event system.

**Extensibility**: To add a new monitoring component (M3 experiment), add
one file to `patternrag/observer/` and register it in `patternrag.yaml`.

---

## Pattern 4 — Factory (Pipeline Construction)

**Location**: `patternrag/factory/`

**Intent**: Define an interface for creating a family of related objects
(retriever, decorators, observers, pipeline) from a configuration, without
specifying concrete classes in the caller.

```
configs/experiment.yaml ──┐
configs/patternrag.yaml ──┴─→  PipelineFactory.build()
                                │
                                ├── 1. Instantiate RetrieverStrategy
                                │      (bm25 | dense | hybrid)
                                │
                                ├── 2. Apply decorators (innermost first)
                                │      logging → caching (outermost)
                                │
                                ├── 3. Instantiate observers
                                │      LatencyLogger, MetricsCollector, CostMonitor
                                │
                                ├── 4. Create EventDispatcher, register observers
                                │
                                ├── 5. Instantiate LLMGenerator
                                │
                                └── 6. Return PatternRAGPipeline
```

**IMPORTANT**: Factory runs ONCE before any query is processed.
It is NOT called during `pipeline.run()`.

**Monolithic equivalent**: Instantiation is done explicitly in
`run_monolithic.py`; no factory class.

---

## Data Flow Comparison

### Monolithic Runtime Query Flow

```
MonolithicRAGPipeline.run(query, k)
  │
  ├── check _cache dict
  ├── BM25Index.search(query, k)           → bm25_docs
  ├── embedder.encode(query)               → query_vec
  ├── FAISSIndex.search(query_vec, k)      → dense_docs
  ├── _rrf_merge(bm25_docs, dense_docs, k) → merged_docs   [inline]
  ├── logger.debug(...)                                     [inline]
  ├── build_prompt(query, merged_docs)
  ├── generator.generate(prompt)           → answer
  ├── store in _cache
  └── return result dict
```

### PatternRAG Runtime Query Flow

```
PatternRAGPipeline.run(query, k)
  │
  ├── dispatcher.notify(QUERY_RECEIVED)     → observers (side-channel)
  │
  ├── retriever.retrieve(query, k)          ← decorated strategy
  │   └── CachingDecorator
  │         └── LoggingDecorator
  │               └── HybridRetriever
  │                     ├── BM25Retriever → BM25Index.search()
  │                     ├── DenseRetriever → FAISSIndex.search()
  │                     └── _rrf_merge() → scored_docs
  │
  ├── dispatcher.notify(RETRIEVAL_DONE)     → observers (side-channel)
  │   ├── LatencyLogger.on_event()
  │   ├── MetricsCollector.on_event()
  │   └── CostMonitor.on_event() [no-op for this event]
  │
  ├── build_prompt(query, scored_docs)
  ├── generator.generate(prompt) → answer
  │
  ├── dispatcher.notify(GENERATION_DONE)    → observers (side-channel)
  │   ├── LatencyLogger.on_event()
  │   └── CostMonitor.on_event()
  │
  └── return result dict
```

---

## Retrieval Equivalence Guarantee

For the same query, k, and loaded indexes:

```
MonolithicRAGPipeline._rrf_merge(bm25_results, dense_results, k)
==
HybridRetriever.retrieve(query, k)
```

Both use **identical RRF implementations** (same formula, same `rrf_k=60`).
This is verified by `tests/test_retrieval_equivalence.py` across 6 test queries,
checking both document ID order and exact RRF score values.

---

## Module Dependency Rules

```
shared/        ← imported by both monolithic/ and patternrag/
monolithic/    ← imports from shared/ only; NEVER from patternrag/
patternrag/    ← imports from shared/ only; NEVER from monolithic/
```

A cross-import between `monolithic/` and `patternrag/` is a bug.
