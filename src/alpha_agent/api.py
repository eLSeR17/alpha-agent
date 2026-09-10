"""HTTP API server for AlphaAgent.

Exposes the guarded agent over FastAPI so it can be used by other
services, web UIs, or mobile apps:

    POST /ask      — ask a financial question
    GET  /health   — liveness probe
    GET  /tools    — list registered tools

Backend selection:
- ``ALPHA_AGENT_BACKEND=ollama`` (default) uses the local Ollama model.
- ``ALPHA_AGENT_BACKEND=openai`` uses an OpenAI-compatible API.

Optional caching (``ALPHA_AGENT_CACHE=1``) answers repeated queries from
a SQLite store, saving latency and tokens.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, status
import httpx
from pydantic import BaseModel, Field

from . import AlphaAgent, GuardedAlphaAgent, TOOL_REGISTRY
from .agent import AlphaAgent as _AlphaAgent
from .cache import ResponseCache
from .llm import OllamaClient, DEFAULT_BASE_URL, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------------

BACKEND = os.getenv("ALPHA_AGENT_BACKEND", "ollama").lower()
CACHE_ENABLED = os.getenv("ALPHA_AGENT_CACHE", "0").lower() in {"1", "true", "yes"}
CACHE_PATH = os.getenv("ALPHA_AGENT_CACHE_PATH", "cache.sqlite3")
CACHE_TTL = float(os.getenv("ALPHA_AGENT_CACHE_TTL", "3600"))

AI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL") or "gpt-4o-mini"
AI_BASE_URL = os.getenv("AI_BASE_URL")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_agent_singleton: GuardedAlphaAgent | None = None
_cache_singleton: ResponseCache | None = None


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


def _clear_singletons() -> None:
    """Reset module-level state (used by tests)."""
    global _agent_singleton, _cache_singleton
    _agent_singleton = None
    _cache_singleton = None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


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


class HealthResponse(BaseModel):
    status: str = "ok"
    backend: str = BACKEND
    model: str = ""
    cache_enabled: bool = CACHE_ENABLED


class ToolsResponse(BaseModel):
    tools: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaAgent API",
    version="1.0.0",
    description="Financial research agent with tool use and guardrails.",
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — also reports the active backend and model."""
    agent = build_agent()
    llm = agent.agent.llm if hasattr(agent.agent, "llm") else None
    return HealthResponse(
        status="ok",
        backend=BACKEND,
        model=getattr(llm, "model", "unknown"),
        cache_enabled=CACHE_ENABLED,
    )


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


@app.post("/ask", response_model=QueryResponse, tags=["agent"])
def ask(req: QueryRequest) -> QueryResponse:
    """Run the guarded agent on a financial question.

    Guards apply automatically:
    - prompt-injection / financial disclaimer (input/output)
    - ticker validation and tool rate limiting
    - anti-hallucination grounding of numbers in the answer
    """
    if not req.question.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="question is empty")

    cache = get_cache()

    # -- cache hit path --
    if cache is not None:
        cached = cache.get(req.question.strip())
        if cached is not None:
            return _to_response(cached, from_cache=True)

    # -- run the agent --
    agent = build_agent()
    try:
        response = agent.analyze(req.question.strip())
    except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
        logger.error("LLM backend unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The LLM backend is unavailable. Check that Ollama is running "
                "and reachable at OLLAMA_BASE_URL (or set ALPHA_AGENT_BACKEND=openai "
                "with OPENAI_API_KEY)."
            ),
        ) from exc

    if cache is not None:
        cache.set(response)

    return _to_response(response, from_cache=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(response: Any, from_cache: bool) -> QueryResponse:
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
    )
