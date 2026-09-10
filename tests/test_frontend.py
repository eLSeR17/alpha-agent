"""Tests for the web UI and API endpoints with auth + rate limiting.

Deterministic: the agent is scripted with a fake, no network/LLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture()
def open_client(monkeypatch, tmp_path):
    """TestClient in auth=none mode with a scripted fake agent."""
    import alpha_agent.api as api_module

    monkeypatch.setattr(api_module, "AUTH_MODE", "none")
    monkeypatch.setattr(api_module, "CACHE_ENABLED", False)
    monkeypatch.setattr(api_module, "SESSIONS_DB_PATH", str(tmp_path / "sessions.sqlite3"))

    fake = MagicMock()
    fake.analyze.side_effect = lambda q, history=None: _agent_response(q)

    api_module._clear_singletons()
    with patch.object(api_module, "build_agent", return_value=fake):
        yield TestClient(api_module.app, raise_server_exceptions=False)


@pytest.fixture()
def key_client(monkeypatch, tmp_path):
    """TestClient in api_key auth mode with one valid key."""
    import alpha_agent.api as api_module

    monkeypatch.setattr(api_module, "AUTH_MODE", "api_key")
    monkeypatch.setattr(api_module, "CACHE_ENABLED", False)
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MIN", 5)
    monkeypatch.setattr(api_module, "AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(api_module, "SESSIONS_DB_PATH", str(tmp_path / "sessions.sqlite3"))

    # create a key in the real store
    from alpha_agent.auth import APIKeyStore

    store = APIKeyStore(db_path=str(tmp_path / "auth.sqlite3"))
    _, secret = store.create_key("test", rate_limit_per_min=5)
    store.close()

    fake = MagicMock()
    fake.analyze.side_effect = lambda q, history=None: _agent_response(q)

    api_module._clear_singletons()
    with patch.object(api_module, "build_agent", return_value=fake):
        client = TestClient(api_module.app, raise_server_exceptions=False)
        client._test_key = secret
        yield client


class TestWebUI:
    def test_index_served(self, open_client) -> None:
        resp = open_client.get("/")
        assert resp.status_code == 200
        assert "AlphaAgent" in resp.text

    def test_static_css(self, open_client) -> None:
        resp = open_client.get("/static/style.css")
        assert resp.status_code == 200
        assert "--accent" in resp.text

    def test_static_js(self, open_client) -> None:
        resp = open_client.get("/static/app.js")
        assert resp.status_code == 200
        assert "ask" in resp.text


class TestSessions:
    def test_create_session(self, open_client) -> None:
        resp = open_client.post("/sessions")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_ask_with_session_returns_session_id(self, open_client) -> None:
        resp = open_client.post("/ask", json={"question": "AAPL price?"})
        # no session id given → single-shot, returns None
        assert resp.json()["session_id"] is None

    def test_ask_with_unknown_session_404(self, open_client) -> None:
        resp = open_client.post("/ask", json={"question": "AAPL?", "session_id": "unknown"})
        assert resp.status_code == 404

    def test_ask_with_valid_session(self, open_client) -> None:
        sid = open_client.post("/sessions").json()["session_id"]
        resp = open_client.post("/ask", json={"question": "AAPL price?", "session_id": sid})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus(self, open_client) -> None:
        open_client.get("/health")  # populate some counters
        resp = open_client.get("/metrics")
        assert resp.status_code == 200
        assert "alpha_agent_requests_total" in resp.text


class TestAuth:
    def test_missing_key_401(self, key_client) -> None:
        resp = key_client.post("/ask", json={"question": "AAPL?"})
        assert resp.status_code == 401

    def test_valid_key_200(self, key_client) -> None:
        resp = key_client.post(
            "/ask",
            json={"question": "AAPL?"},
            headers={"X-API-Key": key_client._test_key},
        )
        assert resp.status_code == 200
        assert "319.97" in resp.json()["answer"]

    def test_rate_limit_exceeded(self, key_client) -> None:
        import alpha_agent.api as api_module
        # force a tiny bucket to trigger 429 quickly
        limiter = api_module.get_ratelimiter()
        key_id = api_module.get_auth().validate_key(key_client._test_key)
        for _ in range(5):
            limiter.check(key_id, per_minute=5)
        resp = key_client.post(
            "/ask",
            json={"question": "AAPL?"},
            headers={"X-API-Key": key_client._test_key},
        )
        assert resp.status_code == 429
