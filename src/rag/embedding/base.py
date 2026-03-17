"""EmbeddingProvider Protocol — structural subtyping for embedding backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Any object implementing these methods is a valid embedding provider."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts (search_document input_type).

        Args:
            texts: List of document text strings to embed.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            EmbeddingError: if the embedding call fails.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (search_query input_type).

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: if the embedding call fails.
        """
        ...
