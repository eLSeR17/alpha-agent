# AlphaAgent — Operations

Runbook for the API service: deploy, health, data, keys, incidents.

## Quick reference

| What | Command |
|------|---------|
| Run locally | `PYTHONPATH=src uvicorn app:app --host 0.0.0.0 --port 8000` |
| Run with auth | `ALPHA_AGENT_AUTH=api_key PYTHONPATH=src uvicorn ...` |
| Create a key | `PYTHONPATH=src python scripts/create_api_key.py --name <name> [--rate-limit N]` |
| Revoke a key | `PYTHONPATH=src python -c "from alpha_agent.auth import APIKeyStore; APIKeyStore().revoke_key('<key_id>')"` |
| List keys | `PYTHONPATH=src python -c "from alpha_agent.auth import APIKeyStore; print(APIKeyStore().list_keys())"` |
| Health | `curl -s localhost:8000/health` |
| Metrics | `curl -s localhost:8000/metrics` |
| Export OpenAPI | `PYTHONPATH=src python scripts/export_openapi.py` |
| Load test | `PYTHONPATH=src python scripts/load_test.py --concurrency 8 --duration 30` |

## Deploy (Docker)

```bash
docker build -f Dockerfile.api -t alpha-agent-api .
docker compose -f docker-compose.prod.yml up -d
```

The image runs uvicorn with **2 workers** (SQLite + WAL supports concurrent
readers; writers serialize on a 5 s busy timeout — see "Data stores" below).
The container **healthcheck** (`HEALTHCHECK --interval=30s` → `GET /health`)
drives orchestrator readiness; `docker compose ps` shows `(healthy)` when
the API is serving.

Secrets are injected via environment (never baked into the image):
`OPENAI_API_KEY` for the optional cloud backend, and the auth DB is mounted
as a volume (see compose file).

## Data stores (3 SQLite DBs)

All three are plain SQLite files with **WAL journaling** and a **5 s busy
timeout**, set at schema init:

| DB | File (default) | Env override | Contents |
|----|----------------|--------------|----------|
| Auth | `auth.sqlite3` | `ALPHA_AGENT_AUTH_DB` | API key digests (SHA-256), rate budgets, revocation flags |
| Sessions | `sessions.sqlite3` | `ALPHA_AGENT_SESSIONS_DB` | conversation turns (user/assistant) |
| Cache | `cache.sqlite3` | `ALPHA_AGENT_CACHE_PATH` | answer cache (TTL 1 h, single-shot only) |

Backup (hot, WAL-safe):

```bash
sqlite3 auth.sqlite3 ".backup 'backups/auth_$(date +%F).sqlite3'"
sqlite3 sessions.sqlite3 ".backup 'backups/sessions_$(date +%F).sqlite3'"
# cache is disposable — drop it to purge TTL answers
```

Restore: stop the container, replace the file, start. Journal files
(`-wal`, `-shm`) are recreated automatically.

## API key lifecycle (recommended)

1. **Create**: `scripts/create_api_key.py --name <owner-service> --rate-limit <budget>`.
   Capture the plaintext immediately — it is shown once.
2. **Rotate**: create a replacement key, switch clients, then revoke the old
   one. Migration is additive — no downtime.
3. **Revoke**: `revoke_key('<key_id>')` → `validate_key` fails from that
   moment; existing buckets remain (harmless).
4. **Audit**: `list_keys()` returns id/name/rate/created/active — no
   digests, no plaintext.

## Incident response

| Symptom | Likely cause | Response |
|---------|--------------|----------|
| `503` on /ask | Ollama down, or wrong `OLLAMA_BASE_URL` | `docker ps` · check Ollama reachable · verify env |
| `429` storm from one client | key budget too small / leak | revoke key, issue new one with proper budget |
| `401` after rotation | client still using old key | confirm revoke + client header |
| `Database is locked` | >1 writer contention | SQLite writer serializes on 5 s busy timeout; use compose (2 workers, WAL) and retry |
| Slow /ask, healthy :500 | LLM latency (see `llm_request_seconds` histogram) | raise `MAX_ITERATIONS` carefully; consider streaming backend |

## Capacity (measured)

`scripts/load_test.py` measures the API with a **scripted fake agent**
(no LLM) to isolate service-plane capacity from LLM latency, and with the
real backend for end-to-end numbers. See `docs/LOAD_TEST.md` for the
current budget. Point: rate-limit budgets should be **below** measured
capacity, not above it.
