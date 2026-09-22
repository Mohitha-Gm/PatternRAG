"""
GoF Decorator Pattern — RetrieverDecorator base class.

The Decorator pattern allows cross-cutting concerns (caching, logging,
query rewriting, etc.) to be added to any RetrieverStrategy without
modifying the concrete strategy classes.

RetrieverDecorator extends RetrieverStrategy so that decorators are
transparent to the pipeline — the pipeline always operates on a
RetrieverStrategy reference, unaware of whether it is decorated.

Composability::

    CachingDecorator(
        LoggingDecorator(
            HybridRetriever(...)
        )
    )

Each layer adds one concern; no existing class needs modification.
"""
from __future__ import annotations

from abc import ABC

from shared.types import ScoredDoc
from patternrag.strategy.base import RetrieverStrategy


class RetrieverDecorator(RetrieverStrategy, ABC):
    """
    Abstract base class for retriever decorators.

    Holds a reference to the wrapped :class:`~patternrag.strategy.base.RetrieverStrategy`
    and delegates ``retrieve()`` to it by default.  Concrete decorators
    override ``retrieve()`` to add behaviour before and/or after delegation.

    Args:
        wrapped: The inner strategy (or another decorator) to wrap.
    """

    def __init__(self, wrapped: RetrieverStrategy) -> None:
        self._wrapped = wrapped

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        """Default: delegate directly to the wrapped strategy."""
        return self._wrapped.retrieve(query, k)
