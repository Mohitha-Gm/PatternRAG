import os
import math
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.evaluation.ragas_evaluator import evaluate_with_ragas
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

def test_single_sample():
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        print("FAIL: GROQ_API_KEY is not set")
        return

    print("Initializing Groq ChatGroq LLM (openai/gpt-oss-120b) and embeddings...")
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, api_key=groq_api_key)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
        )
    )

    question = "Which actor starred in American Beauty?"
    answer = "Kevin Spacey starred in American Beauty."
    contexts = ["American Beauty is a 1999 American film directed by Sam Mendes and written by Alan Ball, starring Kevin Spacey."]
    ground_truth = "Kevin Spacey"

    print("Running evaluate_with_ragas...")
    scores = evaluate_with_ragas(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        model_name="openai/gpt-oss-120b",
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("Returned scores:", scores)
    print("Types:")
    all_floats = True
    for k, v in scores.items():
        is_float = isinstance(v, float)
        is_nan = math.isnan(v)
        print(f"  {k}: value={v}, type={type(v).__name__}, is_float={is_float}, is_nan={is_nan}")
        if not is_float or is_nan:
            all_floats = False

    print("All four metrics are valid Python floats:", all_floats)

if __name__ == "__main__":
    test_single_sample()
