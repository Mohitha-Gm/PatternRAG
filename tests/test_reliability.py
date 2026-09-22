"""Tests for Groq reliability, offline embedding loading, pacing, and checkpointing."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError, APIConnectionError
import httpx

from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.llm.generator import LLMGenerator, _extract_retry_after
from experiments.runner_utils import evaluate_run


def test_embedder_local_files_only_arg():
    """Verify local_files_only parameter is properly accepted and set."""
    with patch("shared.embedding.embedder.SentenceTransformer") as mock_st:
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        assert embedder.local_files_only is True
        mock_st.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            local_files_only=True,
        )


def test_extract_retry_after_from_header():
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, headers={"retry-after": "3.5"}, request=request)
    exc = RateLimitError("Rate limit", response=response, body=None)
    assert _extract_retry_after(exc) == 3.5


def test_extract_retry_after_from_message():
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, request=request)
    exc = RateLimitError("Please try again in 1.8s", response=response, body=None)
    assert _extract_retry_after(exc) == 1.8


def test_llm_generator_rate_limit_retry_success(monkeypatch):
    """Verify LLMGenerator retries on 429 and succeeds on next attempt."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    generator = LLMGenerator(max_retries=3, initial_backoff=0.01)

    mock_choice = MagicMock()
    mock_choice.message.content = "Answer text"
    mock_success = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5))

    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, headers={"retry-after": "0.01"}, request=request)
    exc = RateLimitError("Rate limit", response=response, body=None)

    with patch.object(generator._client.chat.completions, "create", side_effect=[exc, mock_success]) as mock_create:
        with patch("time.sleep"):  # Avoid actual delay in tests
            ans, usage = generator.generate("prompt")

    assert ans == "Answer text"
    assert usage["prompt_tokens"] == 10
    assert mock_create.call_count == 2


def test_llm_generator_rate_limit_exhausted(monkeypatch):
    """Verify LLMGenerator raises after max_retries attempts."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    generator = LLMGenerator(max_retries=2, initial_backoff=0.01)

    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, headers={}, request=request)
    exc = RateLimitError("Rate limit exceeded", response=response, body=None)

    with patch.object(generator._client.chat.completions, "create", side_effect=exc) as mock_create:
        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                generator.generate("prompt")

    assert mock_create.call_count == 2


def test_evaluate_run_checkpoints_and_records_failures(tmp_path: Path):
    """Verify immediate disk checkpointing, failure logging, and summary preservation."""
    queries = [
        {"id": "q1", "question": "Question 1", "gold_answer": "A1", "supporting_doc_ids": ["d1"]},
        {"id": "q2", "question": "Question 2", "gold_answer": "A2", "supporting_doc_ids": ["d2"]},
        {"id": "q3", "question": "Question 3", "gold_answer": "A3", "supporting_doc_ids": ["d3"]},
    ]

    def mock_pipeline_run(query_str: str, k: int):
        if "Question 2" in query_str:
            raise RuntimeError("Connection dropped on Groq")
        return {
            "retrieved_ids": ["d1"],
            "retrieved_docs": [],
            "answer": f"Predicted answer for {query_str}",
            "retrieval_ms": 10.0,
            "generation_ms": 50.0,
            "total_ms": 60.0,
            "cache_hit": False,
        }

    results = evaluate_run(
        queries=queries,
        pipeline_run_fn=mock_pipeline_run,
        top_k=5,
        run_dir=tmp_path,
        system_name="test_sys",
        skip_ragas=True,
        inter_query_delay=0.0,
    )

    # Check answers.jsonl
    answers_file = tmp_path / "answers.jsonl"
    assert answers_file.exists()
    saved_answers = [json.loads(line) for line in answers_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(saved_answers) == 2
    assert saved_answers[0]["id"] == "q1"
    assert saved_answers[1]["id"] == "q3"

    # Check failed_queries.jsonl
    failed_file = tmp_path / "failed_queries.jsonl"
    assert failed_file.exists()
    saved_failures = [json.loads(line) for line in failed_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(saved_failures) == 1
    assert saved_failures[0]["id"] == "q2"
    assert "Connection dropped on Groq" in saved_failures[0]["error"]

    # Check summary files preserved
    assert (tmp_path / "retrieval_metrics.json").exists()
    assert (tmp_path / "latency.json").exists()
    assert (tmp_path / "failed_summary.json").exists()
    assert results["latency"]["n_queries_completed"] == 2
    assert results["latency"]["n_queries_failed"] == 1
