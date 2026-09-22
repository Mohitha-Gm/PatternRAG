"""
FAISS index — low-level dense vector search infrastructure.

This module provides approximate nearest-neighbour search over document
embeddings using the FAISS library (``faiss-cpu``).

It is ONLY responsible for index construction, serialisation, and search.
Retrieval strategy logic (DenseRetriever) lives in patternrag/strategy/.

Both the monolithic pipeline and PatternRAG's DenseRetriever call this module.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import faiss
import numpy as np

from shared.types import Document, ScoredDoc


class FAISSIndex:
    """
    Wraps a FAISS flat inner-product index over document embeddings.

    We store embeddings normalised to unit length and use inner-product
    search, which is equivalent to cosine similarity for normalised vectors.

    Attributes:
        _corpus:    The list of Documents indexed (order matches _index).
        _index:     FAISS IndexFlatIP instance.
        _dim:       Embedding dimensionality.
    """

    def __init__(self) -> None:
        self._corpus: list[Document] = []
        self._index: faiss.IndexFlatIP | None = None
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(
        self,
        corpus: list[Document],
        embedder,  # SentenceTransformerEmbedder — avoid circular import
    ) -> "FAISSIndex":
        """
        Encode *corpus* documents and build a FAISS flat index.

        Args:
            corpus:   List of :class:`~shared.types.Document` objects.
            embedder: An object with ``encode(texts) -> np.ndarray``.

        Returns:
            self (for chaining).
        """
        self._corpus = corpus
        texts = [doc.text for doc in corpus]

        # Encode in batches; returns (N, D) float32 array.
        embeddings: np.ndarray = embedder.encode(texts)
        embeddings = embeddings.astype(np.float32)

        # Normalise rows so inner-product == cosine similarity.
        faiss.normalize_L2(embeddings)

        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save index and corpus to *path*.

        FAISS index is serialised via faiss.write_index; the corpus and
        dimension are stored in an accompanying ``.meta`` pickle file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path))
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "dim": self._dim}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSIndex":
        """Load a previously saved index from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. "
                "Run  python experiments/build_indexes.py  first."
            )
        meta_path = path.with_suffix(".meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found at '{meta_path}'.")

        idx = cls()
        idx._index = faiss.read_index(str(path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        idx._corpus = meta["corpus"]
        idx._dim = meta["dim"]
        return idx

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredDoc]:
        """
        Return the top-*k* documents by cosine similarity.

        Args:
            query_vector: 1-D or (1, D) float32 numpy array.
            k:            Number of results to return.

        Returns:
            List of :class:`~shared.types.ScoredDoc`, highest score first.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built or loaded.")

        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[ScoredDoc] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS sentinel for not-found
                continue
            results.append(ScoredDoc(doc=self._corpus[idx], score=float(score)))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index is not None else 0
