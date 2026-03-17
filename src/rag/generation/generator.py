"""Cohere Command R+ chat with documents= param for grounded generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import cohere

from rag.core.errors import GenerationError
from rag.core.logging import get_logger
from rag.generation.context import PackedContext

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result from the generation step."""

    answer: str
    citations: list[dict[str, Any]]  # Cohere citation objects
    model: str
    finish_reason: str


async def generate(
    query: str,
    context: PackedContext,
    *,
    api_key: str,
    model: str = "command-r-plus",
) -> GenerationResult:
    """Generate a grounded answer using Cohere Command R+.

    Uses the native `documents=` parameter which causes Command R+ to:
    1. Ground its answer strictly in the provided documents.
    2. Return structured citation objects pointing to specific document spans.
    3. Refuse to answer if the documents are insufficient (controlled by system prompt).

    Args:
        query: Original user query.
        context: Packed context from context.pack_context().
        api_key: Cohere API key.
        model: Cohere chat model.

    Returns:
        GenerationResult with answer text and citations.

    Raises:
        GenerationError: on Cohere API failure.
    """
    client = cohere.Client(api_key=api_key)

    # Build documents list for Cohere's native grounding
    cohere_documents = [
        {
            "id": str(citation.index),
            "text": citation.text,
            "document_id": citation.document_id,
        }
        for citation in context.citations
    ]

    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.chat(
                message=query,
                model=model,
                documents=cohere_documents,
                temperature=0.0,  # deterministic for regulated-industry use
            ),
        )
    except Exception as exc:
        logger.error("generation_failed", model=model, error=str(exc))
        raise GenerationError(
            f"Cohere generation failed: {exc}",
            detail=str(exc),
        ) from exc

    answer = response.text
    citations: list[dict[str, Any]] = []
    if response.citations:
        for c in response.citations:
            citations.append({
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "document_ids": [d.id for d in c.documents] if c.documents else [],
            })

    finish_reason = (
        response.finish_reason if hasattr(response, "finish_reason") else "unknown"
    )

    logger.info(
        "generation_ok",
        model=model,
        answer_len=len(answer),
        citations=len(citations),
        context_tokens=context.token_count,
        finish_reason=finish_reason,
    )

    return GenerationResult(
        answer=answer,
        citations=citations,
        model=model,
        finish_reason=finish_reason,
    )
