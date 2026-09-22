# PatternRAG

**Research project comparing Monolithic vs. GoF-pattern-based RAG architectures.**

> Research question: *"Can applying selected GoF design patterns make a RAG pipeline more extensible and maintainable compared with a monolithic implementation, while preserving comparable RAG performance?"*

## Architecture Overview

```
PatternRAG/
├── shared/          ← Low-level infrastructure (both systems import from here)
├── monolithic/      ← Intentionally flat, all-inline RAG baseline
├── patternrag/      ← Decomposed using four GoF patterns
│   ├── strategy/    ← Strategy Pattern: BM25Retriever, DenseRetriever, HybridRetriever
│   ├── decorator/   ← Decorator Pattern: CachingDecorator, LoggingDecorator
│   ├── observer/    ← Observer Pattern: LatencyLogger, MetricsCollector, CostMonitor
│   └── factory/     ← Factory Pattern: PipelineFactory (construction-time only)
├── experiments/     ← Experiment entry points
├── results/         ← Saved experiment outputs
├── analysis/        ← Comparison and plotting scripts
└── tests/           ← Full test suite (70 tests)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
# Edit .env → set GROQ_API_KEY=<your-key>

# 3. Build indexes (one-time, ~10-30 min depending on hardware)
python experiments/build_indexes.py --config configs/experiment.yaml
```

## Configuration

| File | Purpose |
|---|---|
| `configs/experiment.yaml` | **Master config** — shared by both systems (dataset, LLM, top-k, etc.) |
| `configs/patternrag.yaml` | **PatternRAG wiring** — which retriever/decorators/observers to use |

Hybrid retrieval uses **standard Reciprocal Rank Fusion** (RRF constant k=60). No alpha weight.

## Running Experiments

```bash
# Run both systems under identical conditions (recommended)
python experiments/run_both.py \
    --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml

# Run individual systems
python experiments/run_monolithic.py --config configs/experiment.yaml
python experiments/run_patternrag.py --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml

# Skip RAGAS for faster smoke testing
python experiments/run_both.py --config ... --skip-ragas
```

## Results

Results are saved to `results/{YYYYMMDD_HHMMSS}_{system}/`:

```
results/
└── 20260914_001000_monolithic/
│   ├── config_snapshot.yaml
│   ├── retrieval_metrics.json
│   ├── ragas_scores.json
│   ├── latency.json
│   └── answers.jsonl
└── 20260914_001000_patternrag/
    ├── config_snapshot.yaml
    ├── retrieval_metrics.json
    ├── ragas_scores.json
    ├── latency.json
    ├── answers.jsonl
    └── observer_summaries.json
```

## Comparing Results

```bash
python analysis/compare_results.py --run-id 20260914_001000
python analysis/plots.py --run-id 20260914_001000 --output-dir figures/
```

## Running Tests

```bash
python -m pytest tests/ -v
```

**70 tests** covering: BM25Index, FAISSIndex, retrieval metrics, LatencyTracker, prompt templates, BM25Retriever, DenseRetriever, HybridRetriever, CachingDecorator, LoggingDecorator, PipelineEvent, EventDispatcher, LatencyLogger, MetricsCollector, CostMonitor, PipelineFactory, and the **retrieval equivalence test** (verifies Monolithic RRF == PatternRAG HybridRetriever).

## Maintainability Experiments (M1/M2/M3)

After both systems are stable, perform the three representative modification experiments:

| ID | Task | Metric of interest |
|---|---|---|
| M1 | Add TF-IDF retriever | Files changed, LOC added, existing code modified |
| M2 | Add query-rewriting decorator | Files changed, LOC added, existing code modified |
| M3 | Add hit-rate observer | Files changed, LOC added, existing code modified |

Record results in `analysis/maintainability_notes.md`.

## Research Notes

- **Both systems use identical retrieval algorithms** (BM25, dense, RRF). The difference is architecture only.
- **Retrieval equivalence is verified** by `tests/test_retrieval_equivalence.py`.
- **Both systems use the same underlying retrieval and generation components**; therefore, differences in functional RAG quality are evaluated under controlled conditions, while any runtime overhead introduced by the architectures is measured separately.
- **Do not conclude that the research hypothesis is proven** based on code running correctly. The research conclusion requires the actual functional and architectural experiment results.
