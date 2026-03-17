"""Sentence-boundary sliding-window chunker using tiktoken for token counting."""

from __future__ import annotations

import re
from dataclasses import dataclass

import nltk
import tiktoken

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass(frozen=True)
class Chunk:
    """A text chunk with token count and position metadata."""

    text: str
    token_count: int
    chunk_index: int
    start_sentence: int  # index of first sentence in this chunk
    end_sentence: int    # index of last sentence (exclusive)


def chunk_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split text into overlapping chunks that respect sentence boundaries.

    The algorithm:
    1. Sentence-tokenize with NLTK (respects abbreviations, URLs, etc.)
    2. Greedily add sentences until we would exceed max_tokens
    3. Slide forward, keeping enough sentences to fill overlap_tokens
    4. Repeat until all sentences are covered

    Args:
        text: Input text to chunk.
        max_tokens: Maximum tokens per chunk (inclusive).
        overlap_tokens: Approximate token overlap between adjacent chunks.
        encoding_name: tiktoken encoding to use for token counting.

    Returns:
        List of Chunk objects. If text is empty, returns empty list.
        Guaranteed to cover all tokens in the input (no data loss).
    """
    text = text.strip()
    if not text:
        return []

    enc = tiktoken.get_encoding(encoding_name)
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    # Pre-compute token counts for each sentence
    sentence_tokens = [len(enc.encode(s)) for s in sentences]

    chunks: list[Chunk] = []
    start = 0  # sentence index

    while start < len(sentences):
        current_tokens = 0
        end = start

        # Greedily fill up to max_tokens
        while end < len(sentences) and current_tokens + sentence_tokens[end] <= max_tokens:
            current_tokens += sentence_tokens[end]
            end += 1

        # Edge case: single sentence exceeds max_tokens — include it anyway
        # to avoid infinite loop and ensure complete coverage
        if end == start:
            end = start + 1
            current_tokens = sentence_tokens[start]

        chunk_text_str = " ".join(sentences[start:end])
        chunks.append(
            Chunk(
                text=chunk_text_str,
                token_count=current_tokens,
                chunk_index=len(chunks),
                start_sentence=start,
                end_sentence=end,
            )
        )

        if end >= len(sentences):
            break

        # Slide back to achieve overlap: find how many sentences from the end
        # of this chunk sum to approximately overlap_tokens
        overlap_accumulated = 0
        overlap_start = end
        while overlap_start > start and overlap_accumulated < overlap_tokens:
            overlap_start -= 1
            overlap_accumulated += sentence_tokens[overlap_start]

        # Advance at least one sentence to guarantee progress
        next_start = max(start + 1, overlap_start)
        start = next_start

    return chunks


def _clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()
