"""
Shared dataclasses used by both Monolithic and PatternRAG systems.
These are plain data containers with no architectural opinion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document in the retrieval corpus."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ")
        return f"Document(id={self.id!r}, text={snippet!r}...)"


@dataclass
class ScoredDoc:
    """A retrieved document with its retrieval score."""
    doc: Document
    score: float

    def __repr__(self) -> str:
        return f"ScoredDoc(id={self.doc.id!r}, score={self.score:.4f})"
