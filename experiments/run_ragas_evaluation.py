"""
run_ragas_evaluation.py — Standalone RAGAS evaluation over frozen experiment outputs.

Evaluates generation and context quality without rerunning RAG pipelines or
regenerating answers. Reads the frozen answers.jsonl from:
  - results/20260921_114121_monolithic/answers.jsonl
  - results/20260921_114121_patternrag/answers.jsonl

Resolves contexts from artifacts/corpus.jsonl and runs:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
using Groq (openai/gpt-oss-120b) and all-MiniLM-L6-v2 embeddings.

Outputs are saved to results/20260921_114121_ragas/.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

# Load .env with override to ensure latest key is always active
load_dotenv(override=True)

from shared.evaluation.ragas_evaluator import evaluate_with_ragas, aggregate_ragas_scores
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RUN_ID = "20260921_114121"
JUDGE_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INTER_QUERY_DELAY = 2.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 3.0
REQUIRED_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def is_valid_scores(scores: Any) -> bool:
    """Check if scores dictionary contains valid finite float numbers for all 4 metrics."""
    if not isinstance(scores, dict):
        return False
    for m in REQUIRED_METRICS:
        if m not in scores:
            return False
        v = scores[m]
        if not isinstance(v, (int, float)) or math.isnan(v):
            return False
    return True


def load_existing_checkpoints(checkpoint_file: Path) -> dict[str, dict[str, Any]]:
    """
    Load existing completed records from checkpoint file.
    Returns mapping of query id -> record for valid records only.
    """
    if not checkpoint_file.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("id")
                if qid and is_valid_scores(rec.get("scores")):
                    completed[qid] = rec
            except Exception as e:
                logger.warning("Error parsing line in %s: %s", checkpoint_file, e)
    return completed


def load_corpus_map(corpus_path: Path) -> dict[str, str]:
    """Load corpus mapping doc_id -> doc_text."""
    logger.info("Loading corpus from %s...", corpus_path)
    corpus_map = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus_map[d["id"]] = d["text"]
    logger.info("Loaded %d corpus documents.", len(corpus_map))
    return corpus_map


def evaluate_system_records(
    system_name: str,
    answers_path: Path,
    corpus_map: dict[str, str],
    output_dir: Path,
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> dict[str, Any]:
    """
    Evaluate all 30 frozen query answers for a given system with RAGAS.
    Supports resumption: loads already completed valid queries from checkpoint,
    skips them, processes missing queries, and immediately checkpoints each new result.
    Preserves original query order and separates Monolithic and PatternRAG.
    """
    logger.info("=== Starting RAGAS evaluation for [%s] ===", system_name)
    with open(answers_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    checkpoint_file = output_dir / f"{system_name}_checkpoint.jsonl"
    failed_file = output_dir / f"{system_name}_failed.jsonl"

    # Load existing valid records from checkpoint
    completed_by_id = load_existing_checkpoints(checkpoint_file)
    logger.info("[%s] Found %d valid completed queries in checkpoint.", system_name, len(completed_by_id))

    per_query_scores: list[dict[str, Any]] = []
    failed_evaluations: list[dict[str, Any]] = []
    rate_limit_events: int = 0

    display_system = "PatternRAG" if system_name.lower() == "patternrag" else "Monolithic"

    for i, rec in enumerate(records):
        qid = rec["id"]
        question = rec["question"]
        answer = rec["predicted_answer"]
        gold_answer = rec["gold_answer"]
        retrieved_ids = rec.get("retrieved_doc_ids", [])
        contexts = [corpus_map.get(did, "") for did in retrieved_ids]

        # Check if already completed with valid scores
        if qid in completed_by_id:
            logger.info("[%s] Skipping already completed query %d/30 (id: %s)...", system_name, i + 1, qid)
            existing_rec = completed_by_id[qid]
            existing_rec["index"] = i  # preserve original order index
            per_query_scores.append(existing_rec)
            continue

        print(f"[RAGAS] Starting {display_system} query {i + 1}/{len(records)}: {qid}", flush=True)
        logger.info("[%s] Evaluating query %d/30 (id: %s)...", system_name, i + 1, qid)

        scores = None
        tpd_detected = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # If predicted_answer is empty, handle gracefully
                eval_answer = answer if answer.strip() else "No answer provided."
                scores = evaluate_with_ragas(
                    question=question,
                    answer=eval_answer,
                    contexts=contexts,
                    ground_truth=gold_answer,
                    model_name=JUDGE_MODEL,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                if is_valid_scores(scores):
                    break
                else:
                    logger.warning("[%s] Query %d returned incomplete/NaN scores %s, retrying (%d/%d)...",
                                   system_name, i + 1, scores, attempt, MAX_RETRIES)
                    time.sleep(INITIAL_BACKOFF * attempt)
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    rate_limit_events += 1
                if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                    logger.critical("[%s] Daily Token Quota (TPD) reached: %s. Stopping safely.", system_name, exc)
                    tpd_detected = True
                    break
                logger.warning("[%s] Exception on query %d (attempt %d/%d): %s",
                               system_name, i + 1, attempt, MAX_RETRIES, exc)
                time.sleep(INITIAL_BACKOFF * attempt)

        if is_valid_scores(scores):
            query_result = {
                "id": qid,
                "index": i,
                "question": question,
                "predicted_answer": answer,
                "gold_answer": gold_answer,
                "scores": scores,
            }
            per_query_scores.append(query_result)
            completed_by_id[qid] = query_result
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(query_result) + "\n")
                f.flush()

            completed_count = len(completed_by_id)
            remaining_count = len(records) - completed_count
            print("============================================================", flush=True)
            print("[RAGAS PROGRESS]", flush=True)
            print(f"System: {display_system}", flush=True)
            print(f"Query: {i + 1}/{len(records)}", flush=True)
            print(f"Query ID: {qid}", flush=True)
            print("Status: COMPLETED", flush=True)
            print("Metrics:", flush=True)
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                m_val = scores.get(m)
                if isinstance(m_val, float):
                    print(f"  {m}: {m_val:.4f}", flush=True)
                else:
                    print(f"  {m}: {m_val}", flush=True)
            print("Checkpoint: SAVED", flush=True)
            print(f"Completed: {completed_count}/{len(records)}", flush=True)
            print(f"Remaining: {remaining_count}", flush=True)
            print("============================================================", flush=True)
        else:
            short_err = "Failed after max retries or empty/NaN scores returned"
            fail_record = {
                "id": qid,
                "index": i,
                "question": question,
                "error": short_err,
                "last_scores": scores,
            }
            failed_evaluations.append(fail_record)
            with open(failed_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(fail_record) + "\n")
                f.flush()
            logger.error("[%s] Query %d (id: %s) failed RAGAS evaluation.", system_name, i + 1, qid)

            completed_count = len(completed_by_id)
            remaining_count = len(records) - completed_count
            print("============================================================", flush=True)
            print("[RAGAS PROGRESS]", flush=True)
            print(f"System: {display_system}", flush=True)
            print(f"Query: {i + 1}/{len(records)}", flush=True)
            print(f"Query ID: {qid}", flush=True)
            print("Status: FAILED", flush=True)
            print(f"Reason: {short_err}", flush=True)
            print("Checkpoint: failure recorded", flush=True)
            print(f"Completed: {completed_count}/{len(records)}", flush=True)
            print(f"Remaining: {remaining_count}", flush=True)
            print("============================================================", flush=True)

        if tpd_detected:
            logger.critical("[%s] Execution halted safely due to Groq TPD limit.", system_name)
            break

        # Inter-query pacing
        if i < len(records) - 1 and INTER_QUERY_DELAY > 0:
            time.sleep(INTER_QUERY_DELAY)

    # Preserve original query order
    per_query_scores.sort(key=lambda x: x.get("index", 0))

    # Compute aggregates over valid scores
    valid_score_dicts = [item["scores"] for item in per_query_scores if is_valid_scores(item.get("scores"))]
    aggregate = aggregate_ragas_scores(valid_score_dicts)

    system_result = {
        "system": system_name,
        "n_queries_total": len(records),
        "n_queries_completed": len(per_query_scores),
        "n_queries_failed": len(failed_evaluations),
        "rate_limit_events": rate_limit_events,
        "aggregate": aggregate,
        "per_query": per_query_scores,
        "failed": failed_evaluations,
    }

    result_json_path = output_dir / f"{system_name}_ragas.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(system_result, f, indent=2)

    logger.info("[%s] Completed: %d/30 | Failed: %d | Aggregate: %s",
                system_name, len(per_query_scores), len(failed_evaluations), aggregate)
    return system_result


def main():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    artifacts_dir = base_dir / "artifacts"
    ragas_output_dir = results_dir / f"{RUN_ID}_ragas"
    ragas_output_dir.mkdir(parents=True, exist_ok=True)

    mono_answers = results_dir / f"{RUN_ID}_monolithic" / "answers.jsonl"
    prag_answers = results_dir / f"{RUN_ID}_patternrag" / "answers.jsonl"
    corpus_path = artifacts_dir / "corpus.jsonl"

    if not mono_answers.exists():
        raise FileNotFoundError(f"Missing monolithic answers: {mono_answers}")
    if not prag_answers.exists():
        raise FileNotFoundError(f"Missing patternrag answers: {prag_answers}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    # Load corpus
    corpus_map = load_corpus_map(corpus_path)

    # Pre-initialize shared RAGAS judge and embeddings
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    logger.info("Initializing shared judge LLM (%s) and embeddings (%s)...", JUDGE_MODEL, EMBEDDING_MODEL)
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(model=JUDGE_MODEL, temperature=0.0, api_key=groq_api_key)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
        )
    )

    # Evaluate Monolithic
    mono_result = evaluate_system_records(
        system_name="monolithic",
        answers_path=mono_answers,
        corpus_map=corpus_map,
        output_dir=ragas_output_dir,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    # System transition message
    prag_checkpoint_file = ragas_output_dir / "patternrag_checkpoint.jsonl"
    existing_prag = load_existing_checkpoints(prag_checkpoint_file)

    print("============================================================", flush=True)
    print(f"[RAGAS] MONOLITHIC COMPLETE: {mono_result['n_queries_completed']}/30", flush=True)
    print("[RAGAS] Starting PATTERNRAG", flush=True)
    print(f"[RAGAS] Existing valid PatternRAG checkpoints: {len(existing_prag)}/30", flush=True)
    print("[RAGAS] Resuming from missing queries...", flush=True)
    print("============================================================", flush=True)

    # Evaluate PatternRAG
    prag_result = evaluate_system_records(
        system_name="patternrag",
        answers_path=prag_answers,
        corpus_map=corpus_map,
        output_dir=ragas_output_dir,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    # ── Produce Comparison Summary ─────────────────────────────────
    metrics_list = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    comparison = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "metrics": {},
    }

    csv_rows = [["metric", "monolithic", "patternrag", "difference"]]
    for m in metrics_list:
        m_val = mono_result["aggregate"].get(m, 0.0)
        p_val = prag_result["aggregate"].get(m, 0.0)
        diff = m_val - p_val
        comparison["metrics"][m] = {
            "monolithic": m_val,
            "patternrag": p_val,
            "difference": diff,
        }
        csv_rows.append([m, f"{m_val:.4f}", f"{p_val:.4f}", f"{diff:.4f}"])

    with open(ragas_output_dir / "ragas_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(ragas_output_dir / "ragas_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    # ── Save Config & Metadata ─────────────────────────────────────
    config_snapshot = {
        "evaluation_type": "ragas",
        "frozen_run_id": RUN_ID,
        "monolithic_input": str(mono_answers),
        "patternrag_input": str(prag_answers),
        "judge_model": JUDGE_MODEL,
        "judge_provider": "groq",
        "judge_temperature": 0.0,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_local_files_only": True,
        "inter_query_delay": INTER_QUERY_DELAY,
        "max_retries": MAX_RETRIES,
        "metrics_evaluated": metrics_list,
    }
    with open(ragas_output_dir / "ragas_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    metadata = {
        "experiment": "PatternRAG Final RAGAS Evaluation",
        "frozen_functional_run": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "n_queries": 30,
        "monolithic_completed": mono_result["n_queries_completed"],
        "monolithic_failed": mono_result["n_queries_failed"],
        "patternrag_completed": prag_result["n_queries_completed"],
        "patternrag_failed": prag_result["n_queries_failed"],
        "total_rate_limit_events": mono_result["rate_limit_events"] + prag_result["rate_limit_events"],
    }
    with open(ragas_output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=== RAGAS Evaluation Complete! Output saved to: %s ===", ragas_output_dir)


if __name__ == "__main__":
    main()
