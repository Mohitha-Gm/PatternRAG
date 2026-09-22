"""
run_patternrag.py — Entry point for the PatternRAG architecture.

Loads artifacts, uses PipelineFactory to assemble the PatternRAG pipeline
from configuration, runs the same query subset as the monolithic runner,
evaluates results, and saves them to results/.

Usage:
    python experiments/run_patternrag.py \
        --config configs/experiment.yaml \
        --pattern-config configs/patternrag.yaml
    python experiments/run_patternrag.py --config ... --skip-ragas
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import load_yaml
from patternrag.factory.pipeline_factory import PipelineFactory
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
    pattern_config_path: str,
    skip_ragas: bool = False,
    run_id_prefix: str | None = None,
    num_queries: int | None = None,
) -> Path:
    exp_cfg = load_yaml(config_path)
    pat_cfg = load_yaml(pattern_config_path)
    if num_queries is not None:
        exp_cfg["experiment"]["num_queries"] = num_queries

    logger.info("=== PATTERNRAG ARCHITECTURE ===")

    # ── Load shared artifacts ──────────────────────────────────────────
    logger.info("Loading artifacts...")
    corpus, bm25_index, faiss_index, embedder = load_artifacts(exp_cfg)
    logger.info("Corpus: %d docs | BM25: %d | FAISS: %d",
                len(corpus), bm25_index.size, faiss_index.size)

    # ── Factory: assemble pipeline (construction time only) ────────────
    logger.info("PipelineFactory: assembling pipeline from config...")
    factory = PipelineFactory()
    pipeline = factory.build(
        exp_cfg=exp_cfg,
        pat_cfg=pat_cfg,
        bm25_index=bm25_index,
        faiss_index=faiss_index,
        embedder=embedder,
    )
    logger.info("Pipeline assembled with %d observers.", pipeline._dispatcher.observer_count)

    # ── Load queries ───────────────────────────────────────────────────
    queries = load_queries(exp_cfg)
    logger.info("Query set: %d questions", len(queries))

    # ── Create result directory ────────────────────────────────────────
    run_dir = make_run_dir(exp_cfg["paths"]["results_dir"], "patternrag")
    if run_id_prefix:
        new_dir = run_dir.parent / f"{run_id_prefix}_patternrag"
        run_dir.rename(new_dir)
        run_dir = new_dir

    save_config_snapshot(run_dir, exp_cfg, pat_cfg)

    # ── Run evaluation ─────────────────────────────────────────────────
    top_k = exp_cfg["retrieval"]["top_k"]
    llm_cfg = exp_cfg.get("llm", {})
    inter_query_delay = float(llm_cfg.get("inter_query_delay", 2.0))
    results = evaluate_run(
        queries=queries,
        pipeline_run_fn=pipeline.run,
        top_k=top_k,
        run_dir=run_dir,
        system_name="patternrag",
        skip_ragas=skip_ragas,
        inter_query_delay=inter_query_delay,
    )

    # ── Save observer summaries and event log ──────────────────────────
    observer_summaries = pipeline.get_observer_summaries()
    with open(run_dir / "observer_summaries.json", "w") as f:
        json.dump(observer_summaries, f, indent=2)

    with open(run_dir / "observer_log.jsonl", "w", encoding="utf-8") as f:
        for event in pipeline.get_event_log():
            f.write(json.dumps(event) + "\n")

    logger.info("=== PATTERNRAG COMPLETE | Results: %s ===", run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PatternRAG architecture.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--pattern-config", default="configs/patternrag.yaml")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--run-id-prefix", default=None)
    parser.add_argument("--num-queries", type=int, default=None, help="Override number of queries to run")
    args = parser.parse_args()
    main(args.config, args.pattern_config, args.skip_ragas, args.run_id_prefix, args.num_queries)

