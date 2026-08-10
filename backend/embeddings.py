from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def name(self) -> str: ...


class SentenceTransformerEmbedder:
    """Default embedder: small, fast, local, CPU-friendly pretrained model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self._model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def name(self) -> str:
        return self._model_name


_EMBEDDERS = {"minilm": SentenceTransformerEmbedder}


def get_embedder(name: str = "minilm") -> EmbeddingModel:
    return _EMBEDDERS[name]()
