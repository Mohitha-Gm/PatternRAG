# PatternRAG — Experiment Reproduction Protocol

Follow these steps exactly to reproduce the experimental results.

---

## Prerequisites

- Python 3.10+
- Groq API key with access to `llama-3.3-70b-versatile`
- ~4 GB disk space (HotpotQA download + FAISS index)
- Internet connection for first run (HuggingFace dataset + model download)

---

## Step 0 — Environment Setup

```bash
# Clone / navigate to project
cd PatternRAG

# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Open .env and set: GROQ_API_KEY=gsk_...
```

---

## Step 1 — Build Indexes (One-time)

This downloads HotpotQA (validation split, ~7500 rows), builds the
document corpus, and constructs BM25 and FAISS indexes.

```bash
python experiments/build_indexes.py --config configs/experiment.yaml
```

Expected output:
- `artifacts/corpus.jsonl`  — ~97k document records
- `artifacts/bm25_index.pkl`
- `artifacts/faiss_index.bin` + `faiss_index.meta`

**This step is idempotent.** Re-running skips existing artifacts.
To force rebuild: `--force`

**Estimated time**: 5–30 minutes depending on hardware (FAISS encoding dominates).

---

## Step 2 — Verify Setup

```bash
python -m pytest tests/ -v
```

All 70 tests should pass. The critical test is:

```
tests/test_retrieval_equivalence.py
```

This verifies that Monolithic hybrid retrieval == PatternRAG HybridRetriever
for 6 test queries before you run the full experiment.

---

## Step 3 — Run Both Systems

```bash
python experiments/run_both.py \
    --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml
```

This runs both systems sequentially under the **same** `run_id` prefix.

**Or run individually**:
```bash
python experiments/run_monolithic.py --config configs/experiment.yaml
python experiments/run_patternrag.py \
    --config configs/experiment.yaml \
    --pattern-config configs/patternrag.yaml
```

**Note on RAGAS**: RAGAS evaluation makes additional Groq API calls
(~1 call per query × 4 metrics). For 500 queries this may be slow/costly.
To skip RAGAS and only collect retrieval metrics and latency:
```bash
python experiments/run_both.py --config ... --skip-ragas
```

---

## Step 4 — Inspect Results

```bash
# Side-by-side comparison table
python analysis/compare_results.py --run-id <YYYYMMDD_HHMMSS>

# Export to CSV
python analysis/compare_results.py --run-id <YYYYMMDD_HHMMSS> --export results.csv

# Generate figures
python analysis/plots.py --run-id <YYYYMMDD_HHMMSS> --output-dir figures/
```

Result directories are located at:
```
results/<YYYYMMDD_HHMMSS>_monolithic/
results/<YYYYMMDD_HHMMSS>_patternrag/
```

---

## Step 5 — Architectural Modification Experiments (M1/M2/M3)

Perform these after both systems are stable and the functional evaluation
is complete. Run them from a clean git state.

### M1 — Add a New Retrieval Strategy

1. `git checkout -b experiment/m1`
2. **Monolithic**: Edit `monolithic/pipeline.py` — add a TF-IDF branch inline.
3. `git diff --stat` → record files changed, LOC added/deleted.
4. `git stash` (reset monolithic changes).
5. **PatternRAG**: Create `patternrag/strategy/tfidf_retriever.py`, update `patternrag.yaml`.
6. `git diff --stat` → record files changed, LOC added/deleted.
7. Fill in `analysis/maintainability_notes.md` → M1 section.
8. `git checkout main && git branch -D experiment/m1`

### M2 — Add a New Cross-cutting Concern

Same protocol. Task: add a query-rewriting step.
- Monolithic: inline in `monolithic/pipeline.py`
- PatternRAG: new `patternrag/decorator/rewriting_decorator.py` + `patternrag.yaml`

### M3 — Add a New Monitoring Component

Same protocol. Task: add a hit-rate counter.
- Monolithic: inline counter in `monolithic/pipeline.py`
- PatternRAG: new `patternrag/observer/hitrate_observer.py` + `patternrag.yaml`

---

## Configuration Reference

### `configs/experiment.yaml`

| Key | Default | Description |
|---|---|---|
| `experiment.seed` | 42 | Random seed for deterministic query sampling |
| `experiment.num_queries` | 500 | Number of HotpotQA validation questions |
| `experiment.dataset_split` | validation | HotpotQA split |
| `retrieval.top_k` | 10 | Documents retrieved per query |
| `retrieval.rrf_k` | 60 | RRF constant (standard value) |
| `embedding.model_name` | all-MiniLM-L6-v2 | SentenceTransformer model |
| `llm.provider` | groq | LLM provider |
| `llm.model_name` | llama-3.3-70b-versatile | Groq model |
| `llm.max_tokens` | 256 | Max answer tokens |
| `llm.temperature` | 0.0 | LLM temperature |

### `configs/patternrag.yaml`

| Key | Options | Description |
|---|---|---|
| `retriever.type` | bm25, dense, hybrid | Which RetrieverStrategy to use |
| `decorators[].type` | logging, caching | Decorators applied innermost-first |
| `observers[].type` | latency_logger, metrics_collector, cost_monitor | Observers registered |

---

## Important Research Notes

1. **Do not change `experiment.yaml` between systems.** Both use the same file.
2. **Do not re-build indexes between runs.** Use the same artifact files.
3. **Do not conclude the hypothesis is proven** because the code runs.
   The research conclusion requires the actual experimental results.
4. **Retrieval equivalence**: Before concluding differences in RAG quality,
   confirm that `tests/test_retrieval_equivalence.py` passes on the full corpus.
5. Both systems use the same underlying retrieval and generation components;
   therefore, differences in functional RAG quality are evaluated under controlled
   conditions, while any runtime overhead introduced by the architectures is
   measured separately.
