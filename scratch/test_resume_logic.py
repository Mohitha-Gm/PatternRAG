import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_ragas_evaluation import is_valid_scores, load_existing_checkpoints

def test_resume_logic():
    # 1. Test is_valid_scores
    valid_scores = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.95,
        "context_precision": 0.88,
        "context_recall": 1.0
    }
    nan_scores = {
        "faithfulness": float("nan"),
        "answer_relevancy": 0.95,
        "context_precision": 0.88,
        "context_recall": 1.0
    }
    missing_scores = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.95,
    }
    assert is_valid_scores(valid_scores) is True, "Valid scores should return True"
    assert is_valid_scores(nan_scores) is False, "NaN scores should return False"
    assert is_valid_scores(missing_scores) is False, "Missing scores should return False"
    assert is_valid_scores({}) is False, "Empty scores should return False"
    assert is_valid_scores(None) is False, "None scores should return False"
    print("PASS: is_valid_scores correctly identifies valid finite float scores.")

    # 2. Test reading actual monolithic checkpoint file
    actual_checkpoint = Path("results/20260921_114121_ragas/monolithic_checkpoint.jsonl")
    if actual_checkpoint.exists():
        existing = load_existing_checkpoints(actual_checkpoint)
        print(f"Loaded {len(existing)} valid records from {actual_checkpoint}:")
        for qid, rec in existing.items():
            print(f"  ID: {qid}, Index: {rec.get('index')}, Scores: {rec.get('scores')}")

    print("PASS: Resume logic test completed successfully.")

if __name__ == "__main__":
    test_resume_logic()
