"""
compare_results.py — Side-by-side comparison of Monolithic vs PatternRAG results.

Usage:
    python analysis/compare_results.py --run-id 20260914_001000
    python analysis/compare_results.py --run-id 20260914_001000 --export results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metrics(results_dir: Path, run_id: str, system: str) -> dict:
    run_path = results_dir / f"{run_id}_{system}"
    if not run_path.exists():
        return {}
    metrics = {}
    for fname in ["retrieval_metrics.json", "ragas_scores.json", "latency.json"]:
        fpath = run_path / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            if "aggregate" in data:
                metrics.update(data["aggregate"])
            else:
                metrics.update(data)
    return metrics


def print_comparison(mono: dict, prag: dict) -> None:
    all_keys = sorted(set(list(mono.keys()) + list(prag.keys())))
    col_w = 22
    print(f"\n{'Metric':<30} {'Monolithic':>{col_w}} {'PatternRAG':>{col_w}}")
    print("-" * (30 + col_w * 2 + 2))
    for key in all_keys:
        m_val = mono.get(key, "N/A")
        p_val = prag.get(key, "N/A")
        if isinstance(m_val, float):
            m_str = f"{m_val:.4f}"
        elif isinstance(m_val, dict):
            m_str = str({k: f"{v:.1f}" for k, v in m_val.items()})
        else:
            m_str = str(m_val)
        if isinstance(p_val, float):
            p_str = f"{p_val:.4f}"
        elif isinstance(p_val, dict):
            p_str = str({k: f"{v:.1f}" for k, v in p_val.items()})
        else:
            p_str = str(p_val)
        print(f"{key:<30} {m_str:>{col_w}} {p_str:>{col_w}}")


def main(run_id: str, results_dir: str = "results", export: str | None = None) -> None:
    rd = Path(results_dir)
    mono = load_metrics(rd, run_id, "monolithic")
    prag = load_metrics(rd, run_id, "patternrag")

    if not mono and not prag:
        print(f"No results found for run_id '{run_id}' in '{results_dir}'.")
        return

    print(f"\n=== PatternRAG Comparison | run_id: {run_id} ===")
    print_comparison(mono, prag)

    if export:
        import csv
        all_keys = sorted(set(list(mono.keys()) + list(prag.keys())))
        with open(export, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "monolithic", "patternrag"])
            for key in all_keys:
                writer.writerow([key, mono.get(key, ""), prag.get(key, "")])
        print(f"\nExported to: {export}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--export", default=None, help="Export to CSV path.")
    args = parser.parse_args()
    main(args.run_id, args.results_dir, args.export)
