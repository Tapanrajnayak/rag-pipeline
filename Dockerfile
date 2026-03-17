# ── Stage 1: dependency installer ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.2.15 /uv /usr/local/bin/uv

WORKDIR /app

# Copy manifests first for layer caching
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Sync production dependencies only — no dev tools in the image
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user — uid 1000 is conventional and compatible with most orchestrators
RUN groupadd --gid 1000 rag && \
    useradd --uid 1000 --gid rag --shell /bin/false --no-create-home rag

WORKDIR /app

# Copy virtualenv from builder (no pip, no uv, no shell in final image)
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Make venv the active Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--log-config", "/dev/null"]
