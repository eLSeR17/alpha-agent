# AlphaAgent — Load test (measured 2026-09-10)

`scripts/load_test.py` benchmarks the **service plane** (middleware, auth
stubs, sessions round-trip, metrics, serialization) with the LLM downstream
**stubbed** (`scripts/_loadtest_app.py` replaces `build_agent` with a
scripted fast agent) — capacity testing with a slow, jittery dependency
mocked is standard practice. `--url` targets a real deployment for
end-to-end numbers.

## Result (stubbed downstream, WSL2 host, 6 cores)

```
AlphaAgent API load test
  target       : http://127.0.0.1:8766
  concurrency  : 8
  requests     : 425  (200: 425, errors: 0)
  throughput   : 404.9 req/s
  latency p50  : 17.2 ms
  latency p90  : 27.7 ms
  latency p95  : 34.1 ms
  latency p99  : 56.4 ms
```

## Interpretation

- The service plane sustains **~400 req/s** with single-digit-to-tens-of-ms
  latency before the LLM is even involved — the API layer is not the
  bottleneck.
- End-to-end latency is dominated by the LLM (seconds per ReAct turn on a
  local 7B model). Concurrency still helps: interleaved calls pipeline
  while the model generates.
- **Rate-limit budgets should be set below measured capacity**: a per-key
  limit of 60/min from `docs/OPERATIONS.md` is ~4 orders of magnitude under
  what one node can serve — plenty of headroom, by design.
- Numbers are machine-dependent; re-measure before tuning SLOs.

## Repro

```bash
PYTHONPATH=src python scripts/load_test.py --concurrency 8 --duration 20
# real deployment (includes LLM latency):
PYTHONPATH=src python scripts/load_test.py --url http://localhost:8000 --concurrency 4 --duration 15
```
