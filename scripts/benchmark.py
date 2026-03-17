#!/usr/bin/env python3
"""Benchmark — p50/p95/p99 query latency report.

Usage:
    uv run python scripts/benchmark.py --queries 100 --concurrency 10

Requires:
    - Full stack running (docker-compose up -d)
    - ENG_TOKEN env var with a valid JWT
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx

SAMPLE_QUERIES = [
    "What are the tradeoffs between hybrid and pure semantic search?",
    "How does cross-encoder reranking improve retrieval quality?",
    "What is Reciprocal Rank Fusion and how does it work?",
    "How do RAG systems handle GDPR right to erasure?",
    "What privacy considerations apply to RAG audit logs?",
    "How does Cohere Command R+ handle grounded generation?",
    "What is the difference between dense and sparse retrieval?",
    "How does BM25 compare to neural embedding retrieval?",
]


async def run_query(
    client: httpx.AsyncClient,
    query: str,
    token: str,
    base_url: str,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/v1/query",
            json={"query": query},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "ok": response.status_code == 200,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return {"ok": False, "status": 0, "latency_ms": latency_ms, "error": str(exc)}


async def benchmark(
    num_queries: int,
    concurrency: int,
    base_url: str,
    token: str,
) -> None:
    queries = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(num_queries)]
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:

        async def bounded_query(q: str) -> None:
            async with semaphore:
                r = await run_query(client, q, token, base_url)
                results.append(r)

        print(f"Running {num_queries} queries at concurrency={concurrency}...")
        start = time.monotonic()
        await asyncio.gather(*[bounded_query(q) for q in queries])
        total_s = time.monotonic() - start

    latencies = [r["latency_ms"] for r in results]
    ok_count = sum(1 for r in results if r["ok"])
    error_count = num_queries - ok_count

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]

    print(f"\n{'─' * 40}")
    print(f"  Queries:      {num_queries}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Success:      {ok_count} / {num_queries}")
    print(f"  Errors:       {error_count}")
    print(f"  Total time:   {total_s:.1f}s")
    print(f"  Throughput:   {num_queries / total_s:.1f} req/s")
    print(f"{'─' * 40}")
    print(f"  p50 latency:  {p50:.0f}ms")
    print(f"  p95 latency:  {p95:.0f}ms")
    print(f"  p99 latency:  {p99:.0f}ms")
    print(f"  min latency:  {min(latencies):.0f}ms")
    print(f"  max latency:  {max(latencies):.0f}ms")
    print(f"{'─' * 40}")

    if p95 <= 2000:
        print("  ✓ p95 ≤ 2s — acceptance criterion met")
    else:
        print(f"  ✗ p95 = {p95:.0f}ms — exceeds 2s target")


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="RAG pipeline benchmark")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    args = parser.parse_args()

    token = os.environ.get("ENG_TOKEN", "")
    if not token:
        print("ERROR: Set ENG_TOKEN env var to a valid JWT before benchmarking.")
        raise SystemExit(1)

    asyncio.run(benchmark(args.queries, args.concurrency, args.url, token))


if __name__ == "__main__":
    main()
