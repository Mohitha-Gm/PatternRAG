"""
run_both.py — Run both systems under identical conditions.

Executes monolithic and PatternRAG sequentially using the same
timestamp-based run_id prefix so their result directories are paired.

Usage:
    python experiments/run_both.py \
        --config configs/experiment.yaml \
        --pattern-config configs/patternrag.yaml
    python experiments/run_both.py --config ... --skip-ragas
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_monolithic import main as run_monolithic
from experiments.run_patternrag import main as run_patternrag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    config_path: str,
    pattern_config_path: str,
    skip_ragas: bool = False,
    num_queries: int | None = None,
) -> None:
    # Shared run_id ensures both systems' results are trivially matched.
    run_id_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=== run_both | run_id prefix: %s ===", run_id_prefix)

    mono_dir = run_monolithic(config_path, skip_ragas, run_id_prefix, num_queries)
    prag_dir = run_patternrag(config_path, pattern_config_path, skip_ragas, run_id_prefix, num_queries)

    logger.info("=== Both runs complete ===")
    logger.info("Monolithic results : %s", mono_dir)
    logger.info("PatternRAG results : %s", prag_dir)
    logger.info("To compare: python analysis/compare_results.py --run-id %s", run_id_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run both Monolithic and PatternRAG.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--pattern-config", default="configs/patternrag.yaml")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--num-queries", type=int, default=None, help="Override number of queries to run")
    args = parser.parse_args()
    main(args.config, args.pattern_config, args.skip_ragas, args.num_queries)

