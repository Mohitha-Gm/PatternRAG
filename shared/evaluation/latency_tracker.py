"""
Latency tracker.

A simple context manager for measuring wall-clock time of code blocks.
Used identically by both the monolithic pipeline and PatternRAG.
"""
from __future__ import annotations

import time


class LatencyTracker:
    """
    Context manager that measures elapsed wall-clock time in milliseconds.

    Usage::

        with LatencyTracker() as t:
            do_something()
        print(t.elapsed_ms)

    Attributes:
        elapsed_ms: Wall-clock time in milliseconds between ``__enter__``
                    and ``__exit__``.  Zero until the context exits.
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
