#!/usr/bin/env python3
"""Load-test the AlphaAgent HTTP API.

Measures service-plane capacity (requests/sec, latency percentiles, error
rate) with the LLM downstream **stubbed** — see `_loadtest_app.py`. To
measure a real deployment (with real LLM latency), pass `--url`.

Usage:
    PYTHONPATH=src python scripts/load_test.py --concurrency 8 --duration 20
    PYTHONPATH=src python scripts/load_test.py --url http://localhost:8000 --concurrency 4 --duration 15
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time

import httpx

PORT = 8766


def _spawn_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["ALPHA_AGENT_CACHE"] = "0"
    env["ALPHA_AGENT_AUTH"] = "none"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "scripts._loadtest_app:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        env=env,
    )


async def _worker(client: httpx.AsyncClient, url: str, n: int, results: list[float]) -> None:
    for i in range(n):
        payload = {"question": f"What is the price of symbol {i % 100}?"}
        start = time.perf_counter()
        try:
            r = await client.post(f"{url}/ask", json=payload)
            total = time.perf_counter() - start
            results.append((total, r.status_code))
        except httpx.HTTPError:
            results.append((time.perf_counter() - start, 0))


async def _run(url: str, concurrency: int, duration: float, max_requests: int) -> None:
    results: list[tuple[float, int]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        workers = [
            asyncio.create_task(
                _worker(client, url, max_requests, results)
            )
            for _ in range(concurrency)
        ]
        start = time.monotonic()
        while time.monotonic() - start < duration and len(results) < max_requests:
            await asyncio.sleep(0.05)
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    if not results:
        print("no requests completed — check the server is up")
        raise SystemExit(2)

    lats = sorted(t for t, _ in results)
    ok = sum(1 for _, c in results if c == 200)
    n = len(results)
    rps = n / (time.monotonic() - start)

    def pct(p: float) -> float:
        idx = min(len(lats) - 1, int(len(lats) * p))
        return lats[idx] * 1000

    print("=" * 46)
    print("AlphaAgent API load test")
    print("=" * 46)
    print(f"  target       : {url}")
    print(f"  concurrency  : {concurrency}")
    print(f"  requests     : {n}  (200: {ok}, errors: {n - ok})")
    print(f"  throughput   : {rps:.1f} req/s")
    print(f"  latency p50  : {pct(0.5):.1f} ms")
    print(f"  latency p90  : {pct(0.9):.1f} ms")
    print(f"  latency p95  : {pct(0.95):.1f} ms")
    print(f"  latency p99  : {pct(0.99):.1f} ms")
    print("=" * 46)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load-test the AlphaAgent API")
    parser.add_argument("--url", help="target server (default: spawn a stubbed instance)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=20.0, help="seconds (default 20)")
    parser.add_argument("--requests", type=int, default=500, help="max requests (default 500)")
    args = parser.parse_args()

    url = args.url or f"http://127.0.0.1:{PORT}"
    proc = None
    if not args.url:
        proc = _spawn_server()
        time.sleep(3.0)  # allow boot

    try:
        asyncio.run(_run(url, args.concurrency, args.duration, args.requests))
    except KeyboardInterrupt:
        print("aborted")
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
