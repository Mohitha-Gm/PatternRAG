"""
Corpus builder.

Flattens the HotpotQA context paragraphs into a flat list of
:class:`shared.types.Document` objects suitable for BM25 and FAISS indexing.

Each HotpotQA row contains a ``context`` field with multiple titled paragraphs.
We produce one Document per paragraph.

Document ID format: ``{sanitised_title}_para{position_in_context}``

This ID scheme must remain stable — the retrieval equivalence tests and
evaluation code depend on it.
"""
from __future__ import annotations

from datasets import load_dataset

from shared.types import Document


def build_corpus(dataset_split: str = "validation") -> list[Document]:
    """
    Build the retrieval corpus from all paragraphs in *dataset_split*.

    Args:
        dataset_split: HuggingFace split name ("validation" or "train").

    Returns:
        Deduplicated list of :class:`Document` objects, one per paragraph.
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=dataset_split, trust_remote_code=True)

    seen_ids: set[str] = set()
    documents: list[Document] = []

    for row in dataset:
        context_titles: list[str] = row["context"]["title"]
        context_sentences: list[list[str]] = row["context"]["sentences"]

        for pos, (title, sentences) in enumerate(
            zip(context_titles, context_sentences)
        ):
            doc_id = _make_doc_id(title, pos)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            text = " ".join(sentences).strip()
            documents.append(
                Document(
                    id=doc_id,
                    text=text,
                    metadata={"title": title, "context_position": pos},
                )
            )

    return documents


def _make_doc_id(title: str, position: int) -> str:
    """Return a deterministic Document ID."""
    sanitised = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{sanitised}_para{position}"
