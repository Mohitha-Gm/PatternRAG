"""
DenseRetriever — concrete Strategy for dense vector retrieval.

Implements :class:`~patternrag.strategy.base.RetrieverStrategy` using
the shared :class:`~shared.indexing.faiss_index.FAISSIndex` and
:class:`~shared.embedding.embedder.SentenceTransformerEmbedder`.

The underlying dense retrieval algorithm is identical to what the
monolithic pipeline calls inline.
"""
from __future__ import annotations

from shared.indexing.faiss_index import FAISSIndex
from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy


class DenseRetriever(RetrieverStrategy):
    """
    Dense vector retrieval strategy using FAISS and Sentence Transformers.

    Args:
        faiss_index: A pre-loaded :class:`~shared.indexing.faiss_index.FAISSIndex`.
        embedder:    A :class:`~shared.embedding.embedder.SentenceTransformerEmbedder`.
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedder: SentenceTransformerEmbedder,
    ) -> None:
        self._faiss_index = faiss_index
        self._embedder = embedder

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Retrieve top-*k* documents using dense cosine similarity.

        Args:
            query: Raw query string (will be encoded by the embedder).
            k:     Number of results.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc`.
        """
        query_vec = self._embedder.encode(query)
        return self._faiss_index.search(query_vec, k)
