"""Cohere embed-v3 provider with input_type discipline and efficient batching."""

from __future__ import annotations

import asyncio
from typing import Any

import cohere

from rag.core.errors import EmbeddingError
from rag.core.logging import get_logger

logger = get_logger(__name__)

# Cohere embed API batch size limit
_COHERE_EMBED_BATCH_SIZE = 96


class CohereEmbeddingProvider:
    """Wraps the Cohere Embed v3 API.

    Key design decisions:
    - Enforces correct input_type: 'search_document' for ingest,
      'search_query' for retrieval. Mixing types degrades retrieval quality.
    - Batches efficiently to stay within the 96-item API limit.
    - Runs synchronous Cohere client calls in a thread pool so callers
      can use async/await throughout.
    """

    def __init__(self, api_key: str, model: str = "embed-english-v3.0") -> None:
        self._client = cohere.Client(api_key=api_key)
        self._model = model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts with input_type='search_document'.

        Batches input into chunks of up to 96 and runs batches concurrently.

        Args:
            texts: Document texts to embed.

        Returns:
            List of 1024-dimensional float vectors.

        Raises:
            EmbeddingError: if any batch fails.
        """
        if not texts:
            return []
        batches = _make_batches(texts, _COHERE_EMBED_BATCH_SIZE)
        results = await asyncio.gather(
            *[self._embed_batch(b, "search_document") for b in batches]
        )
        return [vec for batch_vecs in results for vec in batch_vecs]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query with input_type='search_query'.

        Args:
            text: Query text.

        Returns:
            1024-dimensional float vector.

        Raises:
            EmbeddingError: if the call fails.
        """
        vecs = await self._embed_batch([text], "search_query")
        return vecs[0]

    async def _embed_batch(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        """Run a single Cohere embed call in a thread pool.

        Args:
            texts: Batch of texts (max 96).
            input_type: Cohere input_type value.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingError: on API failure.
        """
        loop = asyncio.get_running_loop()
        try:
            response: Any = await loop.run_in_executor(
                None,
                lambda: self._client.embed(
                    texts=texts,
                    model=self._model,
                    input_type=input_type,
                    embedding_types=["float"],
                ),
            )
        except Exception as exc:
            logger.error("cohere_embed_failed", input_type=input_type, error=str(exc))
            raise EmbeddingError(
                f"Cohere embed API call failed: {exc}",
                detail=str(exc),
            ) from exc

        embeddings = response.embeddings.float_
        if embeddings is None or len(embeddings) != len(texts):
            raise EmbeddingError(
                "Cohere returned unexpected embedding count",
                detail=f"expected={len(texts)}, got={len(embeddings) if embeddings else 0}",
            )

        logger.debug(
            "cohere_embed_ok",
            count=len(texts),
            input_type=input_type,
            model=self._model,
        )
        return embeddings  # type: ignore[return-value]


def _make_batches(items: list[str], size: int) -> list[list[str]]:
    """Split a list into batches of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]
