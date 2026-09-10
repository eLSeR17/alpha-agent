"""API tests — deterministic, no network, no real LLM.

The agent backend is patched with a fake that returns scripted
responses, so these tests never call Ollama or OpenAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from alpha_agent.schemas import AgentResponse, ToolCall, ToolResult


def _agent_response(query: str, blocked: bool = False) -> AgentResponse:
    data = {"blocked": blocked, "reason": "blocked by guardrail"} if blocked else {"grounded": True}
    return AgentResponse(
        query=query,
        final_answer="I'm sorry, but I cannot process this request." if blocked else "AAPL is trading at 319.97 USD.",
        reasoning=[],
        tool_calls=[] if blocked else [
            ToolResult(
                tool_call=ToolCall(name="get_stock_price", arguments={"symbol": "AAPL"}),
                output='{"symbol": "AAPL", "price": 319.97}',
                success=True,
            )
        ],
        iterations_used=0 if blocked else 2,
        data=data,
    )


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a scripted fake agent (no LLM calls)."""
    fake_guarded = MagicMock()
    fake_guarded.analyze.side_effect = lambda q, history=None: _agent_response(q)
    fake_guarded.agent.llm = MagicMock(model="fake-model")

    import alpha_agent.api as api_module

    with (
        patch.object(api_module, "build_agent", return_value=fake_guarded),
        patch("alpha_agent.api.CACHE_ENABLED", False),
    ):
        yield TestClient(api_module.app)


class TestHealth:
    def test_health_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["cache_enabled"] is False


class TestAsk:
    def test_ask_returns_answer(self, client) -> None:
        resp = client.post("/ask", json={"question": "What is AAPL price?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "319.97" in body["answer"]
        assert body["grounded"] is True
        assert body["from_cache"] is False
        assert len(body["tool_calls"]) == 1
        assert body["tool_calls"][0]["tool_call"]["name"] == "get_stock_price"

    def test_ask_empty_question_422(self, client) -> None:
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_blank_question_422(self, client) -> None:
        resp = client.post("/ask", json={"question": "   "})
        assert resp.status_code == 422

    def test_ask_too_long_question_422(self, client) -> None:
        resp = client.post("/ask", json={"question": "a" * 2001})
        assert resp.status_code == 422


class TestTools:
    def test_lists_tools(self, client) -> None:
        resp = client.get("/tools")
        assert resp.status_code == 200
        body = resp.json()
        names = {t["name"] for t in body["tools"]}
        assert "get_stock_price" in names
        assert "get_stock_history" in names
        assert "search_news" in names
        assert "calculate_risk_metrics" in names
        assert "get_company_info" in names
