# PatternRAG

**A research study comparing a Monolithic RAG implementation with a GoF design-pattern-based RAG architecture.**

> **Research Question:** *"Can applying selected GoF design patterns make a RAG pipeline more extensible and maintainable compared with a monolithic implementation, while preserving comparable RAG performance?"*

PatternRAG is a research project comparing a monolithic RAG implementation with a GoF design-pattern-based RAG architecture. It decomposes the standard Retrieval-Augmented Generation (RAG) pipeline using four Gang of Four (GoF) design patterns:
- **Strategy** — Interchangeable retrieval strategies (BM25, Dense, Hybrid RRF, TF-IDF).
- **Decorator** — Composable, transparent cross-cutting post-processing (Caching, Logging, Result Filtering).
- **Observer** — Decoupled telemetry and event monitoring side-channels (Latency, Retrieval Metrics, Token Costs, Query Complexity).
- **Factory** — Centralized, configuration-driven assembly of pipeline components at construction time.

> *Note:* The application of GoF patterns is designed to address architectural maintainability, extensibility, and change localization; they do not inherently improve RAG accuracy or execution speed.

---

## Architecture Overview

```
PatternRAG/
├── shared/          ← Low-level infrastructure (indexes, embeddings, types, shared by both systems)
├── monolithic/      ← Intentionally flat, all-inline procedural RAG baseline
├── patternrag/      ← Decomposed using four GoF design patterns
│   ├── strategy/    ← Strategy Pattern: BM25Retriever, DenseRetriever, HybridRetriever, TFIDFRetriever
│   ├── decorator/   ← Decorator Pattern: CachingDecorator, LoggingDecorator, ResultFilterDecorator
│   ├── observer/    ← Observer Pattern: LatencyLogger, MetricsCollector, CostMonitor, QueryComplexityMonitor
│   └── factory/     ← Factory Pattern: PipelineFactory (construction-time only)
├── experiments/     ← Experiment runners and evaluation harnesses
├── results/         ← Saved experiment outputs and frozen benchmark artifacts
├── analysis/        ← Comparison tables and visualization scripts
└── tests/           ← Full test suite (97 tests)
```

---

## Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
# Edit .env → set GROQ_API_KEY=<your-key>

# 3. Build retrieval indexes (one-time setup over HotpotQA corpus)
python experiments/build_indexes.py --config configs/experiment.yaml
```

---

## Configuration

| File | Purpose |
|---|---|
| `configs/experiment.yaml` | **Master config** — shared by both systems (dataset, LLM model/temperature, top-k, paths) |
| `configs/patternrag.yaml` | **PatternRAG wiring** — declares active retriever strategy, decorator chain, and registered observers |

Hybrid retrieval in both architectures utilizes standard **Reciprocal Rank Fusion (RRF)** with constant $k=60$ ($score(d) = \sum \frac{1}{60 + rank(d)}$), with no alpha weighting.

---

## Running Experiments

```bash
# Run both systems under identical conditions (recommended)
python experiments/run_both.py \
    --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml

# Run individual pipelines
python experiments/run_monolithic.py --config configs/experiment.yaml
python experiments/run_patternrag.py --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml

# Skip RAGAS evaluation for quick functional verification
python experiments/run_both.py --config configs/experiment.yaml --skip-ragas
```

---

## Running Tests

Execute the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

**97 tests covering core retrieval, patterns, reliability, and M1-M3 architectural experiments:**
- **Initial Baseline Suite (76 tests):** `BM25Index`, `FAISSIndex`, retrieval metrics (MRR, NDCG, Precision, Recall), `LatencyTracker`, prompt templates, `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `CachingDecorator`, `LoggingDecorator`, `PipelineEvent`, `EventDispatcher`, `LatencyLogger`, `MetricsCollector`, `CostMonitor`, `PipelineFactory`, API retry logic, and the **retrieval equivalence test** (verifies Monolithic RRF == PatternRAG HybridRetriever).
- **After M1 (+5 tests, 81 total):** Shared `TFIDFIndex`, `TFIDFRetriever` strategy, factory wiring, and cross-architectural TF-IDF equivalence.
- **After M2 (+8 tests, 89 total):** `ResultFilterDecorator`, threshold filtering logic, top-1 fallback guarantee, backward compatibility, and cross-architectural filtering equivalence.
- **After M3 (+8 tests, 97 total):** `QueryComplexityMonitor`, event isolation, non-intrusiveness, outlier detection rules, and cross-architectural telemetry equivalence.

---

## Final Experimental Results

The functional evaluation was conducted on a frozen 30-query validation subset from HotpotQA under identical retrieval and LLM generation configurations.

- **Run ID:** `20260921_114121`
- **Dataset:** HotpotQA (validation split)
- **Evaluated Queries:** 30 paired queries (60 total executions)
- **Monolithic Completion:** 30/30 (100%)
- **PatternRAG Completion:** 30/30 (100%)
- **Permanent Failures:** 0
- **Hybrid Retrieval Equivalence:** 100%
- **Same Hybrid document IDs and ranking order:** 30/30

### Retrieval Performance Metrics

| Metric | Monolithic RAG | PatternRAG |
|---|---:|---:|
| MRR | 0.8400 | 0.8400 |
| NDCG@10 | 0.8126 | 0.8126 |
| Precision@1 | 0.7667 | 0.7667 |
| Precision@5 | 0.2400 | 0.2400 |
| Precision@10 | 0.1433 | 0.1433 |
| Recall@1 | 0.3833 | 0.3833 |
| Recall@5 | 0.6000 | 0.6000 |
| Recall@10 | 0.7167 | 0.7167 |

*Interpretation:* PatternRAG reproduced the retrieval behavior of the monolithic implementation exactly for the evaluated 30 queries. All retrieval metrics were identical, and Hybrid RRF returned identical document IDs in identical rank order for all queries.

---

### RAGAS Answer-Quality Evaluation

Answer quality was evaluated using RAGAS (v0.4.3) with Groq `openai/gpt-oss-120b` as the LLM judge and local MiniLM embeddings.

- **Run ID:** `20260921_114121_ragas`
- **Evaluation Scope:** 30 queries per system
- **Monolithic Completion:** 30/30
- **PatternRAG Completion:** 30/30
- **Failed Queries:** 0
- **Rate Limit Handling:** 370 transient TPM 429 retry events encountered during evaluation; all retries resolved successfully via exponential backoff.

| RAGAS Metric | Monolithic RAG | PatternRAG |
|---|---:|---:|
| Faithfulness | 0.4500 | 0.4667 |
| Answer Relevancy | 0.2517 | 0.2777 |
| Context Precision | 0.4220 | 0.3940 |
| Context Recall | 0.6333 | 0.6333 |

*Interpretation:* PatternRAG showed broadly comparable answer-quality behavior. Faithfulness and Answer Relevancy were slightly higher for PatternRAG, Context Recall was identical, and Context Precision was slightly lower. These differences should not be interpreted as evidence that PatternRAG improves or degrades answer quality because the evaluation uses only 30 queries and LLM-based judging can introduce variability. The primary contribution of PatternRAG is architectural extensibility rather than improved RAG quality.

---

### Latency Measurements

Final frozen functional run:

| Stage | Statistic | Monolithic RAG | PatternRAG |
|---|---|---:|---:|
| Retrieval | Mean | 966.19 ms | 1443.11 ms |
| Retrieval | Median | 917.55 ms | 1333.15 ms |
| Retrieval | P95 | 1739.05 ms | 1791.45 ms |
| Generation | Mean | 6497.75 ms | 6657.96 ms |
| Generation | Median | 7121.17 ms | 7121.75 ms |
| Generation | P95 | 10557.57 ms | 10566.55 ms |
| Total | Mean | 7464.06 ms | 8101.30 ms |
| Total | Median | 8543.66 ms | 8297.29 ms |
| Total | P95 | 11258.85 ms | 11800.39 ms |

*Interpretation:* The latency measurements should be interpreted cautiously. The experiment encountered 46 HTTP 429 responses and 45 automatic retries from the Groq API. Therefore, latency is treated as a secondary observation rather than evidence that either architecture is inherently faster or slower. Generation latency dominated the overall runtime.

---

### API Reliability

| Measure | Result |
|---|---:|
| Total paired queries | 60 |
| Monolithic completed | 30/30 |
| PatternRAG completed | 30/30 |
| Permanent failures | 0 |
| HTTP 429 responses | 46 |
| Automatic retries | 45 |
| Final successful generations | 60/60 |

The retry mechanism successfully recovered from transient API rate limiting across all query runs.

---

### Architectural Extensibility Experiments

Three controlled modification experiments were conducted to evaluate change localization and extensibility.

| ID | Modification | Pattern | Result |
|---|---|---|---|
| M1 | Add TF-IDF retrieval | Strategy | New retriever added without modifying the core PatternRAG pipeline |
| M2 | Add result filtering | Decorator | New filtering functionality added without modifying existing retrieval strategies |
| M3 | Add query-complexity monitoring | Observer | New monitoring component added without modifying the core PatternRAG pipeline |

In all three experiments, the core PatternRAG execution pipeline remained unchanged. The monolithic implementation required modification of its central pipeline for each corresponding extension. The Factory was extended where necessary to construct and configure the new components.

| Experiment | PatternRAG Change Localization | Monolithic Change Localization |
|---|---|---|
| M1 – TF-IDF | New Strategy + factory registration | Main pipeline retrieval logic |
| M2 – Result Filter | New Decorator + factory registration | Main pipeline filtering logic |
| M3 – Query Monitor | New Observer + factory registration | Main pipeline monitoring logic |

These experiments provide implementation-based evidence of architectural extensibility and change localization. They are not intended as a universal quantitative measure of maintainability.

---

### Overall Findings

PatternRAG preserved the functional retrieval behavior of the monolithic implementation, achieving 100% Hybrid retrieval equivalence on the evaluated 30-query set and identical retrieval metrics. RAGAS results showed comparable answer-quality measurements. The M1-M3 experiments demonstrated that representative extensions could be introduced through independent Strategy, Decorator, and Observer components while keeping the core PatternRAG pipeline unchanged.

---

## Results Directory Structure

Results are archived in the `results/` directory as frozen artifacts:

```
results/
├── 20260921_114121_monolithic/    ← Frozen functional evaluation run for Monolithic RAG
│   ├── config_snapshot.yaml
│   ├── retrieval_metrics.json
│   ├── latency.json
│   └── answers.jsonl
├── 20260921_114121_patternrag/    ← Frozen functional evaluation run for PatternRAG
│   ├── config_snapshot.yaml
│   ├── retrieval_metrics.json
│   ├── latency.json
│   ├── answers.jsonl
│   └── observer_summaries.json
└── 20260921_114121_ragas/         ← Frozen RAGAS answer-quality evaluation
    ├── ragas_comparison.json
    ├── ragas_comparison.csv
    ├── monolithic_ragas.jsonl
    └── patternrag_ragas.jsonl
```

> **Note:** The functional and RAGAS evaluation results in `results/20260921_114121_*` are frozen artifacts and must not be re-run or modified.

---

## Comparing Results

```bash
python analysis/compare_results.py --run-id 20260921_114121
python analysis/plots.py --run-id 20260921_114121 --output-dir figures/
```

---

## Research Notes

- Both systems use the same underlying retrieval and generation configuration.
- Hybrid retrieval uses standard Reciprocal Rank Fusion (RRF), k=60.
- Retrieval equivalence was verified between Monolithic RAG and PatternRAG.
- Final functional evaluation used 30 paired queries.
- Final RAGAS evaluation used the same 30-query evaluation set.
- PatternRAG preserved comparable functional and answer-quality behavior.
- M1-M3 provide architectural evidence through controlled modifications.
- Latency results are affected by external Groq API rate limiting and should not be interpreted as an architectural performance ranking.
- The study evaluates extensibility and change localization rather than claiming universal maintainability improvements.
