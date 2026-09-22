"""
build_indexes.py — One-time index construction script.

Downloads HotpotQA (validation split), builds the flat document corpus,
constructs BM25 and FAISS indexes, and saves them to the artifacts/ directory.

This script is safely rerunnable: it checks whether artifacts already exist
and skips rebuilding them unless --force is passed.

Usage:
    python experiments/build_indexes.py --config configs/experiment.yaml
    python experiments/build_indexes.py --config configs/experiment.yaml --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path so that 'shared', 'monolithic', 'patternrag' are importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import load_yaml
from shared.data.corpus_builder import build_corpus
from shared.embedding.embedder import SentenceTransformerEmbedder
from shared.indexing.bm25_index import BM25Index
from shared.indexing.faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str, force: bool = False) -> None:
    cfg = load_yaml(config_path)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = artifacts_dir / "corpus.jsonl"
    bm25_path = artifacts_dir / "bm25_index.pkl"
    faiss_path = artifacts_dir / "faiss_index.bin"
    faiss_meta_path = artifacts_dir / "faiss_index.meta"

    # ── Build corpus ───────────────────────────────────────────────────
    if corpus_path.exists() and not force:
        logger.info("Corpus already exists at %s. Loading...", corpus_path)
        from shared.types import Document
        corpus = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus.append(Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {})))
        logger.info("Loaded %d documents from corpus.", len(corpus))
    else:
        logger.info("Building corpus from HotpotQA %s split...", cfg["experiment"]["dataset_split"])
        corpus = build_corpus(cfg["experiment"]["dataset_split"])
        logger.info("Built corpus with %d documents.", len(corpus))
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in corpus:
                f.write(json.dumps({"id": doc.id, "text": doc.text, "metadata": doc.metadata}) + "\n")
        logger.info("Corpus saved to %s", corpus_path)

    # ── Build BM25 index ───────────────────────────────────────────────
    if bm25_path.exists() and not force:
        logger.info("BM25 index already exists at %s. Skipping.", bm25_path)
    else:
        logger.info("Building BM25 index over %d documents...", len(corpus))
        bm25 = BM25Index()
        bm25.build(corpus)
        bm25.save(bm25_path)
        logger.info("BM25 index saved to %s", bm25_path)

    # ── Build FAISS index ──────────────────────────────────────────────
    if faiss_path.exists() and faiss_meta_path.exists() and not force:
        logger.info("FAISS index already exists at %s. Skipping.", faiss_path)
    else:
        model_name = cfg["embedding"]["model_name"]
        logger.info("Loading embedding model: %s", model_name)
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        logger.info("Encoding %d documents (this may take several minutes)...", len(corpus))
        faiss_idx = FAISSIndex()
        faiss_idx.build(corpus, embedder)
        faiss_idx.save(faiss_path)
        logger.info("FAISS index saved to %s", faiss_path)

    logger.info("Index build complete. Artifacts in: %s", artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes for PatternRAG.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist.")
    args = parser.parse_args()
    main(args.config, args.force)
