"""
shared package — low-level infrastructure used by both Monolithic and PatternRAG.
This package must NOT contain retrieval strategy logic or GoF pattern classes.
"""
from shared.types import Document, ScoredDoc

__all__ = ["Document", "ScoredDoc"]
