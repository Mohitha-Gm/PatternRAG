"""
Sentence Transformer embedding wrapper.

Thin wrapper around the ``sentence-transformers`` library.  Both the
monolithic pipeline and PatternRAG's DenseRetriever use this class to
produce document and query embeddings.

Using the same model instance (or the same model name) in both systems
is critical for the fair comparison.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """
    Wraps a SentenceTransformer model for encoding text to dense vectors.

    Args:
        model_name: HuggingFace model identifier, e.g.
                    ``"sentence-transformers/all-MiniLM-L6-v2"``.
        device:     ``"cpu"`` or ``"cuda"``.  Defaults to ``"cpu"``.
        batch_size: Batch size used during encoding.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 128,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )

    def encode(
        self,
        texts: Union[str, list[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts to dense float32 vectors.

        Args:
            texts:         A single string or a list of strings.
            show_progress: Whether to display a progress bar for large batches.

        Returns:
            2-D float32 numpy array of shape ``(len(texts), dim)``.
            If a single string is passed, shape is ``(1, dim)``.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # Normalisation done in FAISSIndex
        )
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return self._model.get_sentence_embedding_dimension()
