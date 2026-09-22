"""
GoF Observer Pattern — base classes for the PatternRAG monitoring side-channel.

The Observer pattern decouples monitoring concerns from the main RAG data flow.
The pipeline emits events; observers react to those events independently.

IMPORTANT: Observers are NOT sequential processing stages.  They do not
intercept or modify the main retrieval/generation data flow.  They receive
events via EventDispatcher.notify() after each pipeline stage completes.

Event flow::

    PatternRAGPipeline.run()
        │  emits events at key stages
        ▼
    EventDispatcher.notify(event)
        ├── LatencyLogger.on_event(event)
        ├── MetricsCollector.on_event(event)
        └── CostMonitor.on_event(event)

Defined here:
    - :class:`PipelineEvent` — immutable event data container.
    - :class:`PipelineObserver` — abstract observer interface.
    - :class:`EventDispatcher` — subject that dispatches events to observers.

Standard event types:
    - ``QUERY_RECEIVED``   : fired when run() is called.
    - ``RETRIEVAL_DONE``   : fired after retrieval completes.
    - ``GENERATION_DONE``  : fired after LLM generation completes.
    - ``CACHE_HIT``        : fired when the CachingDecorator returns a cached result.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# PipelineEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineEvent:
    """
    Immutable event emitted by the PatternRAG pipeline.

    Attributes:
        event_type: String identifier, e.g. ``"QUERY_RECEIVED"``.
        payload:    Dict of event-specific data.  Values should be
                    JSON-serialisable where possible.
        timestamp:  Unix timestamp (seconds) when the event was created.
    """
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# PipelineObserver
# ---------------------------------------------------------------------------

class PipelineObserver(ABC):
    """
    Abstract observer interface.

    All concrete observers must implement :meth:`on_event`.  They are
    registered with an :class:`EventDispatcher` and receive events
    asynchronously (in practice synchronously, but logically decoupled).
    """

    @abstractmethod
    def on_event(self, event: PipelineEvent) -> None:
        """
        Handle a pipeline event.

        This method must not raise exceptions; failed handling should be
        logged internally.

        Args:
            event: The :class:`PipelineEvent` to process.
        """
        ...

    def get_summary(self) -> dict[str, Any]:
        """
        Return a summary of data collected by this observer.

        Override in concrete classes to expose accumulated metrics.
        Default returns an empty dict.
        """
        return {}


# ---------------------------------------------------------------------------
# EventDispatcher
# ---------------------------------------------------------------------------

class EventDispatcher:
    """
    Subject in the Observer pattern.

    Maintains a list of registered :class:`PipelineObserver` instances
    and notifies them synchronously when :meth:`notify` is called.

    The pipeline holds a single EventDispatcher instance and calls
    ``notify()`` at key stages.  Observers are registered once at
    construction time (by :class:`~patternrag.factory.pipeline_factory.PipelineFactory`).
    """

    def __init__(self) -> None:
        self._observers: list[PipelineObserver] = []
        self._events: list[PipelineEvent] = []

    def register(self, observer: PipelineObserver) -> None:
        """
        Register an observer to receive future events.

        Args:
            observer: A :class:`PipelineObserver` instance.
        """
        self._observers.append(observer)

    def notify(self, event: PipelineEvent) -> None:
        """
        Dispatch *event* to all registered observers.

        Each observer's ``on_event`` is called in registration order.
        Exceptions in individual observers are caught and logged to
        prevent one observer from breaking others.

        Args:
            event: The event to dispatch.
        """
        self._events.append(event)
        import logging
        _logger = logging.getLogger(__name__)
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Observer %s raised an exception on event %r: %s",
                    type(observer).__name__,
                    event.event_type,
                    exc,
                )

    @property
    def observer_count(self) -> int:
        """Number of registered observers."""
        return len(self._observers)

    def get_events(self) -> list[PipelineEvent]:
        """Return list of all dispatched events."""
        return list(self._events)

