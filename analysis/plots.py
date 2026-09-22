"""
plots.py — Generate paper figures from experiment results.

Usage:
    python analysis/plots.py --run-id 20260914_001000 --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main(run_id: str, results_dir: str = "results", output_dir: str = "figures") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rd = Path(results_dir)
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    def load(system: str, fname: str) -> dict:
        p = rd / f"{run_id}_{system}" / fname
        if not p.exists():
            return {}
        with open(p) as f:
            data = json.load(f)
        return data.get("aggregate", data)

    mono_r = load("monolithic", "retrieval_metrics.json")
    prag_r = load("patternrag", "retrieval_metrics.json")
    mono_l = load("monolithic", "latency.json")
    prag_l = load("patternrag", "latency.json")

    # ── Figure 1: Retrieval metrics bar chart ──────────────────────────
    r_keys = [k for k in mono_r if k in prag_r]
    if r_keys:
        x = np.arange(len(r_keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, [mono_r[k] for k in r_keys], width, label="Monolithic", color="#4C72B0")
        ax.bar(x + width / 2, [prag_r[k] for k in r_keys], width, label="PatternRAG", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(r_keys, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Retrieval Quality Metrics | run_id: {run_id}")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(od / "retrieval_metrics.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {od / 'retrieval_metrics.png'}")

    # ── Figure 2: Latency comparison ───────────────────────────────────
    latency_labels = ["retrieval", "generation", "total"]
    mono_lats = [mono_l.get(s, {}).get("mean_ms", 0) for s in latency_labels]
    prag_lats = [prag_l.get(s, {}).get("mean_ms", 0) for s in latency_labels]
    if any(mono_lats) or any(prag_lats):
        x = np.arange(len(latency_labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, mono_lats, width, label="Monolithic", color="#4C72B0")
        ax.bar(x + width / 2, prag_lats, width, label="PatternRAG", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l.capitalize()} Latency" for l in latency_labels])
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title(f"Latency Comparison | run_id: {run_id}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(od / "latency_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {od / 'latency_comparison.png'}")

    print("Plot generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    main(args.run_id, args.results_dir, args.output_dir)
