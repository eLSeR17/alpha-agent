"""Unit tests for the SQLite response cache."""

from __future__ import annotations

from alpha_agent.cache import ResponseCache
from alpha_agent.schemas import AgentResponse, ToolResult, ToolCall


def _sample_response(query: str = "What is AAPL price?") -> AgentResponse:
    return AgentResponse(
        query=query,
        final_answer="AAPL is trading at 319.97 USD.",
        reasoning=["thought"],
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


class TestCache:
    def test_miss_then_hit(self) -> None:
        cache = ResponseCache(db_path=":memory:")
        assert cache.get("What is AAPL price?") is None
        cache.set(_sample_response())
        cached = cache.get("What is AAPL price?")
        assert cached is not None
        assert cached.final_answer == "AAPL is trading at 319.97 USD."
        assert cached.tool_calls[0].tool_call.name == "get_stock_price"
        cache.close()

    def test_different_query_is_miss(self) -> None:
        cache = ResponseCache(db_path=":memory:")
        cache.set(_sample_response())
        assert cache.get("What is MSFT price?") is None
        cache.close()

    def test_ttl_expiry(self) -> None:
        cache = ResponseCache(db_path=":memory:", ttl_seconds=-1)
        cache.set(_sample_response())
        assert cache.get("What is AAPL price?") is None
        cache.close()

    def test_roundtrip_preserves_all_fields(self) -> None:
        cache = ResponseCache(db_path=":memory:")
        original = _sample_response()
        cache.set(original)
        cached = cache.get(original.query)
        assert cached is not None
        assert cached.query == original.query
        assert cached.reasoning == original.reasoning
        assert cached.iterations_used == original.iterations_used
        assert cached.data == original.data
        assert cached.tool_calls[0].output == original.tool_calls[0].output
        assert cached.tool_calls[0].success is True
        cache.close()
