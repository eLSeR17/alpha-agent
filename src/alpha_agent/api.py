"""HTTP API server for AlphaAgent.

Exposes the guarded agent over FastAPI so it can be used by other
services, web UIs, or mobile apps:

    POST /ask      — ask a financial question (multi-turn via session_id)
    GET  /health   — liveness probe
    GET  /tools    — list registered tools
    GET  /metrics  — Prometheus-style metrics
    POST /sessions — create a conversation session
    GET  /         — web chat UI

Backend selection:
- ``ALPHA_AGENT_BACKEND=ollama`` (default) uses the local Ollama model.
- ``ALPHA_AGENT_BACKEND=openai`` uses an OpenAI-compatible API.

Auth modes (``ALPHA_AGENT_AUTH``):
- ``none`` (default for local dev) — open access
- ``api_key`` — require ``X-API-Key`` header, keys created with
  ``scripts/create_api_key.py``

Other production features: per-key rate limiting, multi-turn sessions,
request IDs, JSON logging, and /metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import TOOL_REGISTRY, AlphaAgent, GuardedAlphaAgent
from .auth import APIKeyStore
from .cache import ResponseCache
from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient
from .metrics import Metrics
from .ratelimit import RateLimiter
from .sessions import SessionStore

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (machine-parseable)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "json_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    root = logging.getLogger("alpha_agent")
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


_configure_logging()

# ---------------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------------

BACKEND = os.getenv("ALPHA_AGENT_BACKEND", "ollama").lower()
AUTH_MODE = os.getenv("ALPHA_AGENT_AUTH", "none").lower()
CACHE_ENABLED = os.getenv("ALPHA_AGENT_CACHE", "0").lower() in {"1", "true", "yes"}
CACHE_PATH = os.getenv("ALPHA_AGENT_CACHE_PATH", "cache.sqlite3")
CACHE_TTL = float(os.getenv("ALPHA_AGENT_CACHE_TTL", "3600"))
RATE_LIMIT_PER_MIN = int(os.getenv("ALPHA_AGENT_RATE_LIMIT", "60"))
AUTH_DB_PATH = os.getenv("ALPHA_AGENT_AUTH_DB", "auth.sqlite3")
SESSIONS_DB_PATH = os.getenv("ALPHA_AGENT_SESSIONS_DB", "sessions.sqlite3")

AI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL") or "gpt-4o-mini"
AI_BASE_URL = os.getenv("AI_BASE_URL")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

# ---------------------------------------------------------------------------
# Agent/service factory (singletons)
# ---------------------------------------------------------------------------

_agent_singleton: GuardedAlphaAgent | None = None
_cache_singleton: ResponseCache | None = None
_auth_singleton: APIKeyStore | None = None
_sessions_singleton: SessionStore | None = None
_ratelimiter_singleton: RateLimiter | None = None
_metrics_singleton: Metrics | None = None


def build_agent() -> GuardedAlphaAgent:
    """Build (or reuse) the guarded agent with the configured backend."""
    global _agent_singleton
    if _agent_singleton is not None:
        return _agent_singleton

    tools = list(TOOL_REGISTRY.values())

    if BACKEND == "openai" and AI_API_KEY:
        from .llm_openai import OpenAIClient

        llm = OpenAIClient(
            api_key=AI_API_KEY,
            model=AI_MODEL,
            base_url=AI_BASE_URL,
        )
        logger.info("Using OpenAI backend: %s", AI_MODEL)
    else:
        llm = OllamaClient(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        logger.info("Using Ollama backend: %s", OLLAMA_MODEL)

    agent = AlphaAgent(llm_client=llm, tools=tools, max_iterations=MAX_ITERATIONS)
    _agent_singleton = GuardedAlphaAgent(agent)
    return _agent_singleton


def get_cache() -> ResponseCache | None:
    """Return the shared cache (creates it on first use if enabled)."""
    global _cache_singleton
    if not CACHE_ENABLED:
        return None
    if _cache_singleton is None:
        _cache_singleton = ResponseCache(db_path=CACHE_PATH, ttl_seconds=CACHE_TTL)
    return _cache_singleton


def get_auth() -> APIKeyStore | None:
    """Return the shared APIKeyStore when auth is enabled, else ``None``."""
    global _auth_singleton
    if AUTH_MODE != "api_key":
        return None
    if _auth_singleton is None:
        _auth_singleton = APIKeyStore(db_path=AUTH_DB_PATH)
    return _auth_singleton


def get_sessions() -> SessionStore:
    """Return the shared SessionStore."""
    global _sessions_singleton
    if _sessions_singleton is None:
        _sessions_singleton = SessionStore(db_path=SESSIONS_DB_PATH)
    return _sessions_singleton


def get_ratelimiter() -> RateLimiter:
    """Return the shared RateLimiter."""
    global _ratelimiter_singleton
    if _ratelimiter_singleton is None:
        _ratelimiter_singleton = RateLimiter(default_per_minute=RATE_LIMIT_PER_MIN)
    return _ratelimiter_singleton


def get_metrics() -> Metrics:
    """Return the shared Metrics registry."""
    global _metrics_singleton
    if _metrics_singleton is None:
        _metrics_singleton = Metrics()
    return _metrics_singleton


def _clear_singletons() -> None:
    """Reset module-level state (used by tests)."""
    global _agent_singleton, _cache_singleton, _auth_singleton
    global _sessions_singleton, _ratelimiter_singleton, _metrics_singleton
    _agent_singleton = None
    _cache_singleton = None
    _auth_singleton = None
    _sessions_singleton = None
    _ratelimiter_singleton = None
    _metrics_singleton = None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Conversation session id (optional)")


class SessionCreateResponse(BaseModel):
    session_id: str


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultInfo(BaseModel):
    tool_call: ToolCallInfo
    output: str = ""
    success: bool = True
    error: str | None = None


class QueryResponse(BaseModel):
    answer: str
    grounded: bool = True
    blocked: bool = False
    block_reason: str | None = None
    tool_calls: list[ToolResultInfo] = Field(default_factory=list)
    iterations_used: int = 0
    from_cache: bool = False
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    backend: str = BACKEND
    model: str = ""
    cache_enabled: bool = CACHE_ENABLED
    auth_mode: str = AUTH_MODE


class ToolsResponse(BaseModel):
    tools: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaAgent API",
    version="2.0.0",
    description="Financial research agent with tool use, guardrails, and multi-turn chat.",
)

# Serve the web chat UI (static/ dir inside the package)
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> Any:
    """Serve the web chat UI."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        from fastapi.responses import FileResponse

        return FileResponse(index_path)
    return JSONResponse(
        {"detail": "Web UI not found — run the server from the repo root (PYTHONPATH=src)."},
        status_code=404,
    )


# ---------------------------------------------------------------------------
# Middleware: request ID + timing + metrics + logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Any:
    request_id = uuid.uuid4().hex[:12]
    start = time.monotonic()
    metrics = get_metrics()
    metrics.inc("requests_total")

    response = await call_next(request)
    if hasattr(response, "headers"):
        response.headers["X-Request-ID"] = request_id

    duration = time.monotonic() - start
    metrics.observe("request_duration_seconds", duration)

    code = getattr(response, "status_code", 500)
    if code >= 400:
        metrics.inc("errors_total")
    if code == 503:
        metrics.inc("backend_unavailable_total")

    logger.info(
        "request",
        extra={
            "json_fields": {
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": code,
                "duration_ms": round(duration * 1000, 2),
                "client": request.client.host if request.client else None,
            }
        },
    )
    return response


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

API_KEY_HEADER = "X-API-Key"


def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency: validate API key when auth mode is enabled."""
    if AUTH_MODE != "api_key":
        return None  # open mode — no auth required
    auth = get_auth()
    if auth is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="auth store unavailable")
    key_id = auth.validate_key(x_api_key or "")
    if key_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return key_id


# ---------------------------------------------------------------------------
# Rate limiting (applied per key id)
# ---------------------------------------------------------------------------


def check_rate_limit(key_id: str | None) -> None:
    """Enforce the per-key rate limit; raise 429 when exhausted."""
    if AUTH_MODE != "api_key" or key_id is None:
        return
    auth = get_auth()
    limit = auth.rate_limit_for(key_id) if auth else RATE_LIMIT_PER_MIN
    limiter = get_ratelimiter()
    allowed, _ = limiter.check(key_id, per_minute=limit)
    if not allowed:
        get_metrics().inc("rate_limited_total")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"X-RateLimit-Remaining": "0"},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — also reports backend, model, auth mode and cache status."""
    agent = build_agent()
    llm = agent.agent.llm if hasattr(agent.agent, "llm") else None
    return HealthResponse(
        status="ok",
        backend=BACKEND,
        model=getattr(llm, "model", "unknown"),
        cache_enabled=CACHE_ENABLED,
        auth_mode=AUTH_MODE,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> str:
    """Prometheus text-format metrics."""
    return get_metrics().render()


@app.get("/tools", response_model=ToolsResponse, tags=["ops"])
def tools() -> ToolsResponse:
    """List the tools the agent can use (name + description)."""
    return ToolsResponse(
        tools=[
            {
                "name": t.name,
                "description": t.description,
                "parameters": [p.model_dump() for p in t.parameters],
            }
            for t in TOOL_REGISTRY.values()
        ]
    )


@app.post("/sessions", response_model=SessionCreateResponse, tags=["sessions"])
def create_session(_: str | None = None) -> SessionCreateResponse:
    """Create a new multi-turn conversation session."""
    return SessionCreateResponse(session_id=get_sessions().create_session())


@app.post("/ask", response_model=QueryResponse, tags=["agent"])
def ask(req: QueryRequest, key_id: str | None = Depends(require_api_key)) -> QueryResponse:
    """Run the guarded agent on a financial question.

    Multi-turn: pass ``session_id`` to continue a conversation.  Without
    it, the question is answered standalone.

    Guards apply automatically — prompt-injection / financial disclaimer
    (input/output), ticker validation and tool rate limiting, and
    anti-hallucination grounding of numbers in the answer.
    """
    metrics = get_metrics()
    check_rate_limit(key_id)

    if not req.question.strip():
        metrics.inc("bad_input_total")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="question is empty")

    sessions = get_sessions()
    session_id = req.session_id
    if session_id and not sessions.exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session {session_id} not found",
        )

    history = sessions.load_messages(session_id) if session_id else []

    # -- cache hit path (standalone queries only — cached answers are
    #    context-free by design) --
    cache = get_cache()
    if cache is not None and not session_id:
        cached = cache.get(req.question.strip())
        if cached is not None:
            metrics.inc("cache_hits_total")
            return _to_response(cached, from_cache=True, session_id=None)

    # -- run the agent --
    agent = build_agent()
    try:
        with metrics.time("llm_request_seconds"):
            response = agent.analyze(req.question.strip(), history=history)
    except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
        logger.error("LLM backend unavailable: %s", exc)
        metrics.inc("backend_unavailable_total")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The LLM backend is unavailable. Check that Ollama is running "
                "and reachable at OLLAMA_BASE_URL (or set ALPHA_AGENT_BACKEND=openai "
                "with OPENAI_API_KEY)."
            ),
        ) from exc

    if cache is not None and not session_id:
        cache.set(response)

    if session_id:
        sessions.append_turn(session_id, req.question.strip(), response.final_answer)

    return _to_response(response, from_cache=False, session_id=session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(response: Any, from_cache: bool, session_id: str | None) -> QueryResponse:
    data = response.data if hasattr(response, "data") else {}
    return QueryResponse(
        answer=response.final_answer,
        grounded=data.get("grounded", True),
        blocked=bool(data.get("blocked", False)),
        block_reason=data.get("reason"),
        tool_calls=[
            ToolResultInfo(
                tool_call=ToolCallInfo(name=tr.tool_call.name, arguments=tr.tool_call.arguments),
                output=tr.output,
                success=tr.success,
                error=tr.error,
            )
            for tr in response.tool_calls
        ],
        iterations_used=response.iterations_used,
        from_cache=from_cache,
        session_id=session_id,
    )
