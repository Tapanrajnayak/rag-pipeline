"""Jinja2 prompt templates — auditable, testable, never f-string soup."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

# StrictUndefined raises an error if a template variable is missing.
# This catches template bugs at render time, not at answer-generation time.
_env = Environment(
    undefined=StrictUndefined,
    autoescape=False,  # plain text, not HTML
    trim_blocks=True,
    lstrip_blocks=True,
)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = _env.from_string("""\
You are a precise, grounded research assistant.
Answer the user's question using ONLY the information in the provided documents.
If the documents do not contain enough information to answer the question, say so clearly.
Do not invent facts, speculate, or use knowledge outside the provided documents.
Always cite the document(s) you used by referencing their index numbers in square brackets, e.g. [1], [2].
""")

# ── Document context block ────────────────────────────────────────────────────

_DOCUMENT_BLOCK_TEMPLATE = _env.from_string("""\
{% for doc in documents %}
[{{ doc.index }}] {{ doc.text }}
{% endfor %}
""")

# ── Query prompt ──────────────────────────────────────────────────────────────

_QUERY_PROMPT_TEMPLATE = _env.from_string("""\
Based on the documents above, answer the following question:
{{ query }}
""")


def render_system_prompt() -> str:
    """Render the system prompt (no variables)."""
    return _SYSTEM_PROMPT_TEMPLATE.render()


def render_document_block(documents: list[dict[str, str]]) -> str:
    """Render the context document block.

    Args:
        documents: List of dicts with 'index' (int) and 'text' (str) keys.

    Returns:
        Formatted document block string.
    """
    return _DOCUMENT_BLOCK_TEMPLATE.render(documents=documents)


def render_query_prompt(query: str) -> str:
    """Render the query prompt.

    Args:
        query: User's question.

    Returns:
        Formatted query string.
    """
    return _QUERY_PROMPT_TEMPLATE.render(query=query)
