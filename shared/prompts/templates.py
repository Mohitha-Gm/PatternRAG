"""
Shared prompt templates.

This module contains the SINGLE prompt template used by BOTH the monolithic
pipeline and PatternRAG.  Using identical prompts is required for a fair
functional comparison.

Do NOT modify this template for one system without updating the other.
"""
from __future__ import annotations

from shared.types import ScoredDoc


_PROMPT_TEMPLATE = """\
Use the following context to answer the question. Be concise and factual.
If the answer cannot be determined from the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, scored_docs: list[ScoredDoc]) -> str:
    """
    Build the RAG prompt from a question and a list of retrieved documents.

    Documents are concatenated in rank order (highest score first).
    Each document is preceded by its rank number.

    Args:
        question:    The user question.
        scored_docs: Retrieved documents ordered by score (desc).

    Returns:
        Formatted prompt string ready for the LLM.
    """
    context_parts: list[str] = []
    for rank, sd in enumerate(scored_docs, start=1):
        context_parts.append(f"[{rank}] {sd.doc.text}")

    context = "\n\n".join(context_parts)
    return _PROMPT_TEMPLATE.format(context=context, question=question)
