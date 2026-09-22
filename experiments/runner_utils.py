"""
Shared experiment runner utilities.

Used by run_monolithic.py and run_patternrag.py to ensure identical
evaluation logic, result schemas, and file I/O.
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from shared.data.hotpotqa_loader import load_hotpotqa
from shared.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    aggregate_retrieval_metrics,
)
from shared.evaluation.latency_tracker import LatencyTracker
from shared.types import Document, ScoredDoc

logger = logging.getLogger(__name__)


def load_artifacts(cfg: dict[str, Any]):
    """
    Load pre-built corpus, BM25 index, FAISS index, and embedder.

    Returns:
        (corpus, bm25_index, faiss_index, embedder)
    """
    from shared.indexing.bm25_index import BM25Index
    from shared.indexing.faiss_index import FAISSIndex
    from shared.embedding.embedder import SentenceTransformerEmbedder

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    corpus_path = artifacts_dir / "corpus.jsonl"
    bm25_path = artifacts_dir / "bm25_index.pkl"
    faiss_path = artifacts_dir / "faiss_index.bin"

    for p in [corpus_path, bm25_path, faiss_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                "Run:  python experiments/build_indexes.py --config configs/experiment.yaml"
            )

    # Load corpus
    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus.append(Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {})))

    bm25_index = BM25Index.load(bm25_path)
    faiss_index = FAISSIndex.load(faiss_path)
    model_name = cfg["embedding"]["model_name"]
    local_files_only = cfg.get("embedding", {}).get("local_files_only", False)
    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        local_files_only=local_files_only,
    )

    return corpus, bm25_index, faiss_index, embedder


def load_queries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the deterministic query subset from HotpotQA."""
    exp = cfg["experiment"]
    return load_hotpotqa(
        split=exp["dataset_split"],
        n_samples=exp["num_queries"],
        seed=exp["seed"],
    )


def make_run_dir(results_dir: str, system: str) -> Path:
    """Create and return a timestamped result directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / f"{ts}_{system}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: Path, exp_cfg: dict, pat_cfg: dict | None = None) -> None:
    """Save exact configs used for this run."""
    snapshot = {"experiment": exp_cfg}
    if pat_cfg is not None:
        snapshot["patternrag"] = pat_cfg
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False)


def evaluate_run(
    queries: list[dict[str, Any]],
    pipeline_run_fn,   # callable(query_str, k) → dict with retrieved_ids, answer, latencies
    top_k: int,
    run_dir: Path,
    system_name: str,
    skip_ragas: bool = False,
    inter_query_delay: float = 0.0,
) -> dict[str, Any]:
    """
    Execute the pipeline over all queries and compute all metrics.

    Args:
        queries:           List of query dicts from load_hotpotqa().
        pipeline_run_fn:   Callable that takes (query, k) and returns a result dict.
        top_k:             Retrieval cutoff.
        run_dir:           Where to write result files.
        system_name:       "monolithic" or "patternrag" (for logging).
        skip_ragas:        If True, skip RAGAS evaluation (for fast smoke tests).
        inter_query_delay: Seconds to wait after each successful query.

    Returns:
        Dict of aggregate metrics.
    """
    from shared.evaluation.ragas_evaluator import evaluate_with_ragas, aggregate_ragas_scores

    per_query_retrieval: list[dict] = []
    per_query_ragas: list[dict] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []
    answers_records: list[dict] = []
    failed_queries: list[dict] = []

    answers_path = run_dir / "answers.jsonl"
    failed_path = run_dir / "failed_queries.jsonl"

    def _latency_stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        s = sorted(vals)
        p95_idx = max(0, int(len(s) * 0.95) - 1)
        return {
            "mean_ms": statistics.mean(vals),
            "median_ms": statistics.median(vals),
            "p95_ms": s[p95_idx],
        }

    def _save_summary_files() -> None:
        """Preserve partial or final aggregated results to disk."""
        if per_query_retrieval:
            agg_retrieval = aggregate_retrieval_metrics(per_query_retrieval)
            with open(run_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"aggregate": agg_retrieval, "per_query": per_query_retrieval}, f, indent=2)

        if not skip_ragas and per_query_ragas:
            agg_ragas = aggregate_ragas_scores(per_query_ragas)
            with open(run_dir / "ragas_scores.json", "w", encoding="utf-8") as f:
                json.dump({"aggregate": agg_ragas, "per_query": per_query_ragas}, f, indent=2)

        if retrieval_latencies:
            lat_summary = {
                "retrieval": _latency_stats(retrieval_latencies),
                "generation": _latency_stats(generation_latencies),
                "total": _latency_stats(total_latencies),
                "n_queries_completed": len(retrieval_latencies),
                "n_queries_failed": len(failed_queries),
            }
            with open(run_dir / "latency.json", "w", encoding="utf-8") as f:
                json.dump(lat_summary, f, indent=2)

        if failed_queries:
            with open(run_dir / "failed_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"failed_count": len(failed_queries), "failed_queries": failed_queries},
                    f,
                    indent=2,
                )

    logger.info("[%s] Running over %d queries (pacing: %.1fs)...", system_name, len(queries), inter_query_delay)

    try:
        for i, q in enumerate(queries):
            question = q["question"]
            gold_answer = q["gold_answer"]
            gold_ids = q["supporting_doc_ids"]

            try:
                result = pipeline_run_fn(question, top_k)
            except Exception as exc:
                logger.error("[%s] Query %d (id: %s) failed: %s", system_name, i, q.get("id"), exc)
                fail_record = {
                    "id": q.get("id"),
                    "query_index": i,
                    "question": question,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp": datetime.now().isoformat(),
                }
                failed_queries.append(fail_record)
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fail_record) + "\n")
                    f.flush()
                continue

            retrieved_ids = result["retrieved_ids"]

            # ── Retrieval metrics ──────────────────────────────────────
            r_metrics = compute_retrieval_metrics(retrieved_ids, gold_ids)
            per_query_retrieval.append(r_metrics)

            # ── Latencies ──────────────────────────────────────────────
            retrieval_latencies.append(result["retrieval_ms"])
            generation_latencies.append(result["generation_ms"])
            total_latencies.append(result["total_ms"])

            # ── RAGAS ─────────────────────────────────────────────────
            if not skip_ragas:
                contexts = [sd.doc.text for sd in result["retrieved_docs"]]
                ragas_scores = evaluate_with_ragas(
                    question=question,
                    answer=result["answer"],
                    contexts=contexts,
                    ground_truth=gold_answer,
                )
                per_query_ragas.append(ragas_scores)

            # ── Answer record — immediately saved to disk ──────────────
            record = {
                "id": q["id"],
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": result["answer"],
                "retrieved_doc_ids": retrieved_ids,
                "retrieval_ms": result["retrieval_ms"],
                "generation_ms": result["generation_ms"],
                "cache_hit": result.get("cache_hit", False),
            }
            answers_records.append(record)
            with open(answers_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()

            if (i + 1) % 50 == 0 or (i + 1) == len(queries):
                logger.info("[%s] Progress: %d/%d", system_name, i + 1, len(queries))

            # ── Request pacing: delay between successful LLM calls ────
            if inter_query_delay > 0 and (i < len(queries) - 1):
                time.sleep(inter_query_delay)

    finally:
        _save_summary_files()

    agg_retrieval = aggregate_retrieval_metrics(per_query_retrieval) if per_query_retrieval else {}
    agg_ragas = aggregate_ragas_scores(per_query_ragas) if (not skip_ragas and per_query_ragas) else {}
    latency_summary = {
        "retrieval": _latency_stats(retrieval_latencies),
        "generation": _latency_stats(generation_latencies),
        "total": _latency_stats(total_latencies),
        "n_queries_completed": len(retrieval_latencies),
        "n_queries_failed": len(failed_queries),
    }

    logger.info(
        "[%s] Results saved to %s (completed: %d, failed: %d)",
        system_name, run_dir, len(answers_records), len(failed_queries),
    )
    return {
        "retrieval": agg_retrieval,
        "ragas": agg_ragas,
        "latency": latency_summary,
        "failed": failed_queries,
    }
