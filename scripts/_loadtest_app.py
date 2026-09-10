#!/usr/bin/env python3
"""Fast-path app for load testing — scripted agent (no LLM).

The load test measures the *service plane* (auth, rate limiting, sessions,
metrics, middleware) with downstream mocked, which is how capacity tests
are done when the dependency (here: an LLM) is slow and jittery.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import alpha_agent.api as api_module
from alpha_agent.schemas import AgentResponse, ToolCall, ToolResult


def _agent_response(query: str) -> AgentResponse:
    return AgentResponse(
        query=query,
        final_answer="AAPL is trading at 319.97 USD.",
        reasoning=[],
        tool_calls=[
            ToolResult(
                tool_call=ToolCall(name="get_stock_price", arguments={"symbol": "AAPL"}),
                output='{"symbol": "AAPL", "price": 319.97}',
                success=True,
            )
        ],
        iterations_used=2,
        data={"grounded": True},
    )


def _fast_agent() -> MagicMock:
    fake = MagicMock()
    fake.analyze.side_effect = lambda q, history=None: _agent_response(q)
    return fake


api_module.build_agent = _fast_agent  # type: ignore[assignment]

from alpha_agent.api import app

__all__ = ["app"]
