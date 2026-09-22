"""
RAGAS evaluation wrapper using Groq.

Wraps the RAGAS library for generation quality evaluation.
Used identically by both the monolithic pipeline and PatternRAG.

RAGAS metrics evaluated:
- faithfulness        : Is the answer grounded in the retrieved context?
- answer_relevancy    : Does the answer address the question?
- context_precision   : Are the retrieved documents relevant to the question?
- context_recall      : Do retrieved documents cover the gold answer?

Uses Groq (via LangChain ChatGroq) for LLM evaluations and the shared
SentenceTransformer model for embeddings, removing OpenAI runtime dependency.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
    model_name: str = "openai/gpt-oss-120b",
    llm: Any = None,
    embeddings: Any = None,
) -> dict[str, float]:
    """
    Run RAGAS evaluation for a single question-answer-context tuple.

    Args:
        question:     The user question.
        answer:       The generated answer from the RAG pipeline.
        contexts:     List of retrieved document texts used to generate the answer.
        ground_truth: Gold standard answer (used for context_recall metric).
        model_name:   Groq model identifier for RAGAS evaluation.
        llm:          Optional pre-initialized LangchainLLMWrapper instance.
        embeddings:   Optional pre-initialized LangchainEmbeddingsWrapper instance.

    Returns:
        Dict with RAGAS metric scores (float, typically 0-1).
        Returns empty dict if RAGAS evaluation fails for this sample.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")

        ragas_llm = llm if llm is not None else LangchainLLMWrapper(
            ChatGroq(model=model_name, temperature=0.0, api_key=groq_api_key)
        )
        ragas_embeddings = embeddings if embeddings is not None else LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        )

        sample = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(sample)

        # Groq only supports n=1 generations per request
        answer_relevancy.strictness = 1

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        def _to_float(val: Any) -> float:
            if isinstance(val, (list, tuple)):
                val = val[0] if val else 0.0
            return float(val) if val is not None else 0.0

        return {
            "faithfulness": _to_float(result["faithfulness"]),
            "answer_relevancy": _to_float(result["answer_relevancy"]),
            "context_precision": _to_float(result["context_precision"]),
            "context_recall": _to_float(result["context_recall"]),
        }

    except Exception as exc:
        logger.warning("RAGAS evaluation failed for question %r: %s", question, exc)
        return {}


def aggregate_ragas_scores(
    per_query_scores: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average RAGAS scores across the full query set.

    Args:
        per_query_scores: List of metric dicts from :func:`evaluate_with_ragas`.
                          Entries may be empty (failed evaluations) and are skipped.

    Returns:
        Dict of averaged scores.  Keys present only if at least one sample succeeded.
    """
    valid = [s for s in per_query_scores if s]
    if not valid:
        return {}

    keys = valid[0].keys()
    return {
        key: sum(s.get(key, 0.0) for s in valid) / len(valid)
        for key in keys
    }
