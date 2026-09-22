"""
run_monolithic.py — Entry point for the Monolithic RAG baseline.

Loads artifacts, instantiates MonolithicRAGPipeline, runs the configured
query subset, evaluates results, and saves them to results/.

Usage:
    python experiments/run_monolithic.py --config configs/experiment.yaml
    python experiments/run_monolithic.py --config configs/experiment.yaml --skip-ragas
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import load_yaml
from shared.llm.generator import LLMGenerator
from monolithic.pipeline import MonolithicRAGPipeline
from experiments.runner_utils import (
    load_artifacts,
    load_queries,
    make_run_dir,
    save_config_snapshot,
    evaluate_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    config_path: str,
    skip_ragas: bool = False,
    run_id_prefix: str | None = None,
    num_queries: int | None = None,
) -> Path:
    exp_cfg = load_yaml(config_path)
    if num_queries is not None:
        exp_cfg["experiment"]["num_queries"] = num_queries

    logger.info("=== MONOLITHIC RAG BASELINE ===")

    # ── Load shared artifacts ──────────────────────────────────────────
    logger.info("Loading artifacts...")
    corpus, bm25_index, faiss_index, embedder = load_artifacts(exp_cfg)
    logger.info("Corpus: %d docs | BM25: %d | FAISS: %d",
                len(corpus), bm25_index.size, faiss_index.size)

    # ── Load LLM ──────────────────────────────────────────────────────
    llm_cfg = exp_cfg.get("llm", {})
    generator = LLMGenerator(
        model_name=llm_cfg.get("model_name", "openai/gpt-oss-120b"),
        max_tokens=llm_cfg.get("max_tokens", 256),
        temperature=llm_cfg.get("temperature", 0.0),
        max_retries=llm_cfg.get("max_retries", 5),
        initial_backoff=llm_cfg.get("initial_backoff", 2.0),
    )

    # ── Instantiate pipeline ───────────────────────────────────────────
    pipeline = MonolithicRAGPipeline(
        bm25_index=bm25_index,
        faiss_index=faiss_index,
        embedder=embedder,
        generator=generator,
        config=exp_cfg,
    )

    # ── Load queries ───────────────────────────────────────────────────
    queries = load_queries(exp_cfg)
    logger.info("Query set: %d questions", len(queries))

    # ── Create result directory ────────────────────────────────────────
    run_dir = make_run_dir(exp_cfg["paths"]["results_dir"], "monolithic")
    if run_id_prefix:
        # Rename dir to use shared prefix from run_both.py
        new_dir = run_dir.parent / f"{run_id_prefix}_monolithic"
        run_dir.rename(new_dir)
        run_dir = new_dir

    save_config_snapshot(run_dir, exp_cfg)

    # ── Run evaluation ─────────────────────────────────────────────────
    top_k = exp_cfg["retrieval"]["top_k"]
    inter_query_delay = float(llm_cfg.get("inter_query_delay", 2.0))
    results = evaluate_run(
        queries=queries,
        pipeline_run_fn=pipeline.run,
        top_k=top_k,
        run_dir=run_dir,
        system_name="monolithic",
        skip_ragas=skip_ragas,
        inter_query_delay=inter_query_delay,
    )

    # ── Save monitoring summary ────────────────────────────────────────
    monitoring = pipeline.get_monitoring_summary()
    with open(run_dir / "monitoring_summary.json", "w") as f:
        json.dump(monitoring, f, indent=2)

    logger.info("=== MONOLITHIC COMPLETE | Results: %s ===", run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monolithic RAG baseline.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS evaluation.")
    parser.add_argument("--run-id-prefix", default=None, help="Shared run_id prefix (used by run_both.py).")
    parser.add_argument("--num-queries", type=int, default=None, help="Override number of queries to run")
    args = parser.parse_args()
    main(args.config, args.skip_ragas, args.run_id_prefix, args.num_queries)

