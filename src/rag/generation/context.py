"""Token-budget-aware context packer.

Greedily packs reranked chunks into a context window, truncating at sentence
boundaries when the last chunk would overflow the budget. Returns both the
packed context string and a citation list for grounded answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import nltk
import tiktoken

from rag.retrieval.reranker import RerankResult

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class PackedContext:
    """Result of context packing."""

    text: str                  # concatenated context for the LLM prompt
    citations: list[Citation]  # ordered list of included chunks / sources
    token_count: int           # total tokens in `text`
    truncated: bool            # True if any chunk was truncated


@dataclass
class Citation:
    """Source reference for a packed chunk."""

    index: int         # 1-based citation number in the prompt
    document_id: str
    chunk_id: str
    text: str          # the (possibly truncated) chunk text included


def pack_context(
    results: list[RerankResult],
    *,
    max_tokens: int = 4096,
    encoding_name: str = "cl100k_base",
) -> PackedContext:
    """Greedily pack chunks into the context window.

    Algorithm:
    1. Iterate results in relevance order (highest first).
    2. For each chunk: if it fits entirely, include it.
    3. If the chunk would overflow: truncate at the last complete sentence
       that fits, include the truncation, then stop.
    4. Build citation list in parallel.

    Args:
        results: Reranked chunks, highest relevance first.
        max_tokens: Maximum tokens for the context block.
        encoding_name: tiktoken encoding for token counting.

    Returns:
        PackedContext with text, citations, token count, and truncation flag.
    """
    enc = tiktoken.get_encoding(encoding_name)
    included: list[tuple[str, str, str]] = []  # (chunk_id, document_id, text)
    total_tokens = 0
    truncated = False

    for result in results:
        chunk_text = result.chunk.text
        chunk_tokens = len(enc.encode(chunk_text))

        if total_tokens + chunk_tokens <= max_tokens:
            # Fits entirely
            included.append((
                result.chunk.chunk_id,
                result.chunk.document_id,
                chunk_text,
            ))
            total_tokens += chunk_tokens
        else:
            # Try to fit a sentence-truncated version
            remaining = max_tokens - total_tokens
            truncated_text = _truncate_to_sentences(chunk_text, remaining, enc)
            if truncated_text:
                included.append((
                    result.chunk.chunk_id,
                    result.chunk.document_id,
                    truncated_text,
                ))
                total_tokens += len(enc.encode(truncated_text))
                truncated = True
            break  # budget exhausted

    citations = [
        Citation(
            index=i + 1,
            document_id=doc_id,
            chunk_id=chunk_id,
            text=text,
        )
        for i, (chunk_id, doc_id, text) in enumerate(included)
    ]

    packed_text = "\n\n".join(text for _, _, text in included)
    return PackedContext(
        text=packed_text,
        citations=citations,
        token_count=total_tokens,
        truncated=truncated,
    )


def _truncate_to_sentences(
    text: str,
    max_tokens: int,
    enc: tiktoken.Encoding,
) -> str:
    """Truncate text to the last complete sentence that fits within max_tokens.

    Args:
        text: Input text to truncate.
        max_tokens: Maximum token budget.
        enc: tiktoken encoder.

    Returns:
        Truncated string ending at a sentence boundary, or empty string if
        even the first sentence does not fit.
    """
    sentences = nltk.sent_tokenize(text)
    accumulated: list[str] = []
    token_count = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        if token_count + sentence_tokens > max_tokens:
            break
        accumulated.append(sentence)
        token_count += sentence_tokens

    return " ".join(accumulated)
