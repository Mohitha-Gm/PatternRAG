"""
Retrieval quality metrics.

Implements standard IR evaluation metrics.  Both the monolithic pipeline
and PatternRAG use these functions identically.

All functions operate on document ID lists (strings), not on Document objects,
so they are independent of the retrieval architecture.
"""
from __future__ import annotations

import math


def _reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Return the reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int]) -> float:
    """Compute Discounted Cumulative Gain."""
    return sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
        if rel > 0
    )


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute retrieval quality metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (desc relevance).
        gold_ids:      List of ground-truth relevant document IDs.
        k_values:      k cutoffs for Recall@k and Precision@k.
                       Defaults to [1, 5, 10].

    Returns:
        Dict with keys: Recall@k, Precision@k (for each k), MRR, NDCG@10.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    gold_set = set(gold_ids)
    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in gold_set)
        metrics[f"Recall@{k}"] = hits / len(gold_set) if gold_set else 0.0
        metrics[f"Precision@{k}"] = hits / k if k > 0 else 0.0

    # MRR
    metrics["MRR"] = _reciprocal_rank(retrieved_ids, gold_set)

    # NDCG@10
    k_ndcg = 10
    top_k_ndcg = retrieved_ids[:k_ndcg]
    relevances = [1 if doc_id in gold_set else 0 for doc_id in top_k_ndcg]
    ideal_relevances = sorted(relevances, reverse=True)
    dcg = _dcg(relevances)
    idcg = _dcg(ideal_relevances)
    metrics["NDCG@10"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def aggregate_retrieval_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average per-query metrics across the full query set.

    Args:
        per_query_metrics: List of metric dicts from :func:`compute_retrieval_metrics`.

    Returns:
        Dict of averaged metrics.
    """
    if not per_query_metrics:
        return {}

    keys = per_query_metrics[0].keys()
    return {
        key: sum(m[key] for m in per_query_metrics) / len(per_query_metrics)
        for key in keys
    }
