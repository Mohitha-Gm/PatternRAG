"""
TF-IDF index — low-level keyword search infrastructure.

This module provides TF-IDF document scoring via scikit-learn's TfidfVectorizer.
It is responsible for index construction, serialisation, and cosine similarity search.
Retrieval strategy logic (TFIDFRetriever) lives in patternrag/strategy/.

Both the monolithic pipeline and PatternRAG's TFIDFRetriever call this module.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.types import Document, ScoredDoc


class TFIDFIndex:
    """
    Wraps a scikit-learn TfidfVectorizer index over a fixed corpus.

    Attributes:
        _corpus:     The list of Documents indexed.
        _vectorizer: Underlying TfidfVectorizer instance.
        _matrix:     Document-term sparse matrix.
    """

    def __init__(self) -> None:
        self._corpus: list[Document] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, corpus: list[Document]) -> "TFIDFIndex":
        """
        Build the TF-IDF index from *corpus*.

        Args:
            corpus: List of :class:`~shared.types.Document` objects.

        Returns:
            self (for chaining).
        """
        self._corpus = corpus
        texts = [doc.text for doc in corpus]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            norm="l2",
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise index, vectorizer, matrix, and corpus to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus": self._corpus,
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                },
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TFIDFIndex":
        """Load a previously saved index from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TF-IDF index not found at '{path}'.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._corpus = data["corpus"]
        idx._vectorizer = data["vectorizer"]
        idx._matrix = data["matrix"]
        return idx

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Return the top-*k* documents ranked by TF-IDF cosine similarity.

        Args:
            query: Raw query string.
            k:     Number of results to return.

        Returns:
            List of :class:`~shared.types.ScoredDoc`, highest score first.
        """
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("TF-IDF index has not been built or loaded.")

        if not self._corpus or k <= 0:
            return []

        query_vec = self._vectorizer.transform([query])
        # Compute cosine similarities against all documents
        similarities = cosine_similarity(query_vec, self._matrix).flatten()

        # Sort indices by similarity score in descending order
        num_candidates = min(k, len(self._corpus))
        top_indices = np.argsort(similarities)[::-1][:num_candidates]

        return [
            ScoredDoc(doc=self._corpus[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._corpus)
