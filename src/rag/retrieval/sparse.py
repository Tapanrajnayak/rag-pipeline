"""Sparse (BM25) vector generation via fastembed.

fastembed.SparseTextEmbedding produces Qdrant-compatible sparse vectors
(lists of indices + values) without needing an Elasticsearch sidecar.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastembed import SparseTextEmbedding

from rag.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "Qdrant/bm25"


class SparseEncoder:
    """BM25 sparse encoder using fastembed.

    Instantiation downloads the model on first use (~20MB).
    Subsequent calls are in-process (no network).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SparseTextEmbedding | None = None

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            logger.info("loading_sparse_model", model=self._model_name)
            self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    def encode_single(self, text: str) -> dict[str, Any]:
        """Encode a single text synchronously.

        Args:
            text: Input text.

        Returns:
            Dict with 'indices' (list[int]) and 'values' (list[float]).
        """
        model = self._get_model()
        embeddings = list(model.embed([text]))
        emb = embeddings[0]
        return {
            "indices": emb.indices.tolist(),
            "values": emb.values.tolist(),
        }

    async def encode(self, text: str) -> dict[str, Any]:
        """Async wrapper around encode_single for use in async contexts."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode_single, text)

    async def encode_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Encode a batch of texts asynchronously.

        Args:
            texts: Input texts.

        Returns:
            List of sparse vector dicts.
        """
        loop = asyncio.get_running_loop()

        def _batch() -> list[dict[str, Any]]:
            model = self._get_model()
            result = []
            for emb in model.embed(texts):
                result.append({
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                })
            return result

        return await loop.run_in_executor(None, _batch)
