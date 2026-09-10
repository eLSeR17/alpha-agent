# AlphaAgent HTTP API

Base URL: `http://localhost:8000` (dev) · Auth: none by default, `X-API-Key`
when `ALPHA_AGENT_AUTH=api_key`. Interactive docs: `GET /docs` (Swagger,
auto-generated from the OpenAPI schema — export with
`PYTHONPATH=src python scripts/export_openapi.py`).

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET  | `/`            | – | Web chat UI (vanilla JS) |
| POST | `/ask`         | key* | Ask a question; multi-turn via `session_id` |
| POST | `/sessions`    | key* | Create a conversation session |
| GET  | `/health`      | – | Liveness + backend/model/auth/cache status |
| GET  | `/tools`       | – | Tool registry (name, description, parameters) |
| GET  | `/metrics`     | – | Prometheus text-format metrics |
| GET  | `/docs`        | – | Swagger UI (OpenAPI) |

\* only when `ALPHA_AGENT_AUTH=api_key`.

## POST /ask

Request:

```json
{
  "question": "What is the current price of AAPL?",
  "session_id": "AxT2rJMRljR0sm4XrUnW8g"
}
```

- `question` — required, 1–2000 chars.
- `session_id` — optional. Omit for a single-shot question (cacheable);
  provide to continue a conversation (last 20 turns are injected through
  the guardrails).

Response `200`:

```json
{
  "answer": "AAPL is trading at 319.97 USD.",
  "grounded": true,
  "blocked": false,
  "block_reason": null,
  "tool_calls": [
    {
      "tool_call": {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
      "output": "{\"symbol\": \"AAPL\", \"price\": 319.97}",
      "success": true,
      "error": null
    }
  ],
  "iterations_used": 2,
  "from_cache": false,
  "session_id": "AxT2rJMRljR0sm4XrUnW8g"
}
```

Fields: `answer` (final text) · `grounded` (numbers cross-checked against
tool results) · `blocked` (input guardrail rejected the question) ·
`block_reason` · `tool_calls` (evidence trail) · `iterations_used` (ReAct
loop budget spent) · `from_cache` (SQLite cache hit — single-shot only) ·
`session_id` (echoed when using sessions).

## POST /sessions

```
POST /sessions  →  200 {"session_id": "AxT2rJMRljR0sm4XrUnW8g"}
```

## Error codes

| Code | When | Body `detail` (example) |
|------|------|-------------------------|
| 401  | missing/invalid API key (auth mode) | `"Invalid or missing API key. Provide it in the X-API-Key header."` |
| 404  | unknown `session_id` | `"session <id> not found"` |
| 422  | empty/blank question | `"question is empty"` |
| 429  | per-key rate limit exhausted | `"Rate limit exceeded. Try again shortly."` |
| 503  | LLM backend unreachable | `"The LLM backend is unavailable. Check that Ollama is running and reachable at OLLAMA_BASE_URL (or set ALPHA_AGENT_BACKEND=openai with OPENAI_API_KEY)."` |

`401`/`429` carry headers: `WWW-Authenticate: ApiKey`, `X-RateLimit-Remaining: 0`.
Every response carries `X-Request-ID`.

## Auth quick start

```bash
# 1. create a key (plaintext shown once; only the hash is stored)
PYTHONPATH=src python scripts/create_api_key.py --name alice --rate-limit 30

# 2. run the server in api_key mode
ALPHA_AGENT_AUTH=api_key PYTHONPATH=src uvicorn app:app --host 0.0.0.0 --port 8000

# 3. call with the key
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: aa_..." \
  -d '{"question": "What is the price of AAPL?"}'
```

Revoke a key: `PYTHONPATH=src python -c "from alpha_agent.auth import APIKeyStore; APIKeyStore().revoke_key('aa_...')"`.

## Prometheus metrics (GET /metrics)

```
# TYPE alpha_agent_requests_total counter
# TYPE alpha_agent_request_duration_seconds histogram   (p50/p90/p95/p99)
# TYPE alpha_agent_errors_total counter                (4xx+5xx)
# TYPE alpha_agent_backend_unavailable_total counter   (503s)
# TYPE alpha_agent_rate_limited_total counter          (429s)
# TYPE alpha_agent_cache_hits_total counter
# TYPE alpha_agent_llm_request_seconds histogram       (LLM latency)
```

Point Prometheus at `/metrics`; `X-Request-ID` correlates entries in the
JSON request log with traces.
