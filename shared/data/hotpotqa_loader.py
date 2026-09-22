"""
HotpotQA loader.

Returns a deterministic, seeded subset of questions and their gold
supporting document IDs.  The document IDs here reference entries in
the corpus produced by corpus_builder.build_corpus().

Each returned record has the shape::

    {
        "id":                  str,   # HotpotQA question id
        "question":            str,
        "gold_answer":         str,
        "supporting_doc_ids":  list[str],  # corpus document IDs
        "level":               str,   # "easy" | "medium" | "hard"
        "type":                str,   # "bridge" | "comparison"
    }
"""
from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


def load_hotpotqa(
    split: str = "validation",
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Download (or load from HuggingFace cache) the HotpotQA dataset,
    sample *n_samples* questions deterministically, and return them as
    a list of dicts with stable supporting_doc_ids.

    The supporting_doc_ids use the format  ``{title}_para{idx}``  which
    matches the IDs produced by :func:`shared.data.corpus_builder.build_corpus`.

    Args:
        split:     HuggingFace dataset split name ("validation" or "train").
        n_samples: Number of questions to sample.  If the split has fewer
                   rows, all rows are returned.
        seed:      Random seed for reproducibility.

    Returns:
        List of question dicts.
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    total = len(dataset)
    if n_samples >= total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(total), n_samples)

    records: list[dict[str, Any]] = []
    for idx in indices:
        row = dataset[idx]
        # Build stable supporting_doc_ids matching corpus_builder output.
        sup_ids: list[str] = []
        for title, sent_idx in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
        ):
            # Each supporting fact names a title; we map that to the
            # corpus paragraph document ID.  corpus_builder produces one
            # Document per paragraph (all sentences of a paragraph joined).
            para_id = _make_doc_id(title, row, sent_idx)
            if para_id and para_id not in sup_ids:
                sup_ids.append(para_id)

        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["answer"],
                "supporting_doc_ids": sup_ids,
                "level": row["level"],
                "type": row["type"],
            }
        )

    return records


def _make_doc_id(title: str, row: dict, sent_idx: int) -> str | None:
    """
    Given a HotpotQA row and a supporting fact (title, sent_idx), return
    the corpus Document ID for the paragraph that contains that sentence.

    corpus_builder creates one Document per context entry (paragraph).
    The ID format is  ``{sanitised_title}_para{context_position}``.
    """
    # Find which context entry corresponds to this title.
    context_titles: list[str] = row["context"]["title"]
    try:
        pos = context_titles.index(title)
    except ValueError:
        return None
    sanitised = _sanitise(title)
    return f"{sanitised}_para{pos}"


def _sanitise(title: str) -> str:
    """Remove characters that are unsafe in identifiers / filenames."""
    return title.replace(" ", "_").replace("/", "_").replace("\\", "_")
