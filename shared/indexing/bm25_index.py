"""
BM25 index — low-level keyword search infrastructure.

This module provides BM25 document scoring via the ``rank_bm25`` library.
It is ONLY responsible for index construction, serialisation, and search.
Retrieval strategy logic (BM25Retriever) lives in patternrag/strategy/.

Both the monolithic pipeline and PatternRAG's BM25Retriever call this module.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

from rank_bm25 import BM25Okapi

from shared.types import Document, ScoredDoc


class BM25Index:
    """
    Wraps a BM25Okapi index over a fixed corpus.

    Attributes:
        _corpus:    The list of Documents indexed.
        _bm25:      Underlying BM25Okapi instance.
    """

    def __init__(self) -> None:
        self._corpus: list[Document] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, corpus: list[Document]) -> "BM25Index":
        """
        Build the BM25 index from *corpus*.

        Tokenisation: lowercase whitespace-split (matches standard BM25
        usage; kept simple for reproducibility).

        Args:
            corpus: List of :class:`~shared.types.Document` objects.

        Returns:
            self (for chaining).
        """
        self._corpus = corpus
        tokenised = [doc.text.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenised)
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise index and corpus to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        """Load a previously saved index from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{path}'. "
                "Run  python experiments/build_indexes.py  first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._corpus = data["corpus"]
        idx._bm25 = data["bm25"]
        return idx

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Return the top-*k* documents ranked by BM25 score.

        Args:
            query: Raw query string (tokenised internally).
            k:     Number of results to return.

        Returns:
            List of :class:`~shared.types.ScoredDoc`, highest score first.
        """
        if self._bm25 is None:
            raise RuntimeError("Index has not been built or loaded.")

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Pair each score with its corpus document and sort descending.
        ranked = sorted(
            zip(scores, self._corpus),
            key=lambda x: x[0],
            reverse=True,
        )
        return [ScoredDoc(doc=doc, score=float(score)) for score, doc in ranked[:k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._corpus)
