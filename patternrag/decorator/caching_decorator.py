"""
CachingDecorator — in-memory query-result cache.

Adds query-level caching to any RetrieverStrategy without modifying it.
The cache is keyed on the exact query string and k value.
Cache hits return the stored result and emit a CACHE_HIT event if an
EventDispatcher is available.

This is architecturally equivalent to the plain-dict cache in the
monolithic pipeline, but separated into its own composable class.
"""
from __future__ import annotations

from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy
from patternrag.decorator.base import RetrieverDecorator


class CachingDecorator(RetrieverDecorator):
    """
    Caches retrieval results in memory, keyed by ``(query, k)``.

    On a cache miss, delegates to the wrapped strategy and stores the result.
    On a cache hit, returns the stored result without calling the wrapped strategy.

    Args:
        wrapped: The inner :class:`~patternrag.strategy.base.RetrieverStrategy`.
    """

    def __init__(self, wrapped: RetrieverStrategy) -> None:
        super().__init__(wrapped)
        # Cache: maps (query_str, k) → list[ScoredDoc]
        self._cache: dict[tuple[str, int], list[ScoredDoc]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """
        Return cached results if available; otherwise delegate and cache.

        Args:
            query: User query string.
            k:     Number of results requested.

        Returns:
            Ranked list of :class:`~shared.types.ScoredDoc`.
        """
        cache_key = (query, k)

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1
        results = self._wrapped.retrieve(query, k)
        self._cache[cache_key] = results
        return results

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    @property
    def cache_hits(self) -> int:
        """Total number of cache hits."""
        return self._hits

    @property
    def cache_misses(self) -> int:
        """Total number of cache misses."""
        return self._misses

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Evict all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
