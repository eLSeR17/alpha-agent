"""Unit tests for AlphaAgent financial tools.

All yfinance calls are mocked — no network requests are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alpha_agent.tools.market import get_company_info, get_stock_history, get_stock_price
from alpha_agent.tools.news import search_news
from alpha_agent.tools.risk import calculate_risk_metrics
from alpha_agent.tools import TOOL_REGISTRY


# ======================================================================
# Helpers
# ======================================================================

def _make_fast_info(**kwargs: Any) -> SimpleNamespace:
    """Build a mock ``ticker.fast_info`` object."""
    defaults = {"last_price": 185.50, "previous_close": 183.27, "currency": "USD"}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_history_df(n: int = 22, base_price: float = 180.0) -> pd.DataFrame:
    """Create a realistic mock DataFrame of OHLCV data."""
    import numpy as np

    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    high = close + abs(np.random.randn(n))
    low = close - abs(np.random.randn(n))
    volume = np.random.randint(10_000_000, 80_000_000, size=n)

    return pd.DataFrame({
        "Open": close - 1,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


def _mock_ticker(fast_info: Any = None, info: dict | None = None, history_df: pd.DataFrame | None = None, news: list | None = None) -> MagicMock:
    """Return a mock ``yf.Ticker`` with controlled return values."""
    ticker = MagicMock()
    if fast_info is not None:
        ticker.fast_info = fast_info
    if info is not None:
        ticker.info = info
    if history_df is not None:
        ticker.history.return_value = history_df
    if news is not None:
        ticker.news = news
    return ticker


# ======================================================================
# Tests: get_stock_price
# ======================================================================


class TestStockPrice:
    """get_stock_price returns structured price data."""

    def test_valid_symbol(self) -> None:
        mock_ticker = _mock_ticker(fast_info=_make_fast_info(last_price=190.25, previous_close=188.00))
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_price("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["price"] == 190.25
        assert result["currency"] == "USD"
        assert result["previous_close"] == 188.00
        assert result["change_pct"] is not None
        assert abs(result["change_pct"] - 1.20) < 0.1  # ~1.2%

    def test_price_none_returns_error(self) -> None:
        mock_ticker = _mock_ticker(fast_info=_make_fast_info(last_price=None))
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_price("INVALID")

        assert "error" in result

    def test_network_error_returns_error(self) -> None:
        with patch("alpha_agent.tools.market.yf.Ticker", side_effect=ConnectionError("network down")):
            result = get_stock_price("AAPL")

        assert "error" in result
        assert "network down" in result["error"]

    def test_symbol_uppercased_in_output(self) -> None:
        mock_ticker = _mock_ticker(fast_info=_make_fast_info())
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_price("aapl")

        assert result["symbol"] == "AAPL"


# ======================================================================
# Tests: get_stock_history
# ======================================================================


class TestStockHistory:
    """get_stock_history returns aggregated historical data."""

    def test_valid_period(self) -> None:
        df = _make_history_df(n=22, base_price=180.0)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_history("AAPL", period="1mo")

        assert result["symbol"] == "AAPL"
        assert result["period"] == "1mo"
        assert result["data_points"] == 22
        assert result["high"] > result["low"]
        assert result["avg"] > 0
        assert "start_date" in result
        assert "end_date" in result

    def test_invalid_period(self) -> None:
        result = get_stock_history("AAPL", period="invalid")
        assert "error" in result
        assert "Invalid period" in result["error"]

    def test_empty_history(self) -> None:
        mock_ticker = _mock_ticker(history_df=pd.DataFrame())
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_history("AAPL")

        assert "error" in result

    def test_default_period_is_1mo(self) -> None:
        df = _make_history_df(n=5)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_history("AAPL")

        assert result["period"] == "1mo"


# ======================================================================
# Tests: calculate_risk_metrics
# ======================================================================


class TestRiskMetrics:
    """calculate_risk_metrics produces valid risk numbers."""

    def test_calculates_volatility(self) -> None:
        df = _make_history_df(n=63, base_price=150.0)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL", period="3mo")

        assert "volatility" in result
        assert isinstance(result["volatility"], float)
        assert result["volatility"] >= 0

    def test_calculates_sharpe(self) -> None:
        df = _make_history_df(n=63, base_price=150.0)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL", period="3mo")

        assert "sharpe_ratio" in result
        assert isinstance(result["sharpe_ratio"], float)

    def test_calculates_max_drawdown(self) -> None:
        df = _make_history_df(n=63, base_price=150.0)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL", period="3mo")

        assert "max_drawdown" in result
        assert result["max_drawdown"] <= 0  # drawdown is always <= 0

    def test_invalid_period(self) -> None:
        result = calculate_risk_metrics("AAPL", period="10y")
        assert "error" in result

    def test_insufficient_data(self) -> None:
        mock_ticker = _mock_ticker(history_df=_make_history_df(n=1))
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL", period="1mo")

        assert "error" in result

    def test_includes_metadata(self) -> None:
        df = _make_history_df(n=63)
        mock_ticker = _mock_ticker(history_df=df)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL", period="3mo")

        assert result["symbol"] == "AAPL"
        assert result["period"] == "3mo"
        assert "annualised_return" in result
        assert "data_points" in result
        assert "risk_free_rate" in result


# ======================================================================
# Tests: search_news
# ======================================================================


class TestNews:
    """search_news returns financial news articles."""

    def test_search_returns_results(self) -> None:
        fake_news = [
            {"title": "Apple beats earnings", "publisher": "Reuters", "link": "https://example.com/1", "providerPublishTime": "2025-01-01"},
            {"title": "iPhone sales surge", "publisher": "Bloomberg", "link": "https://example.com/2", "providerPublishTime": "2025-01-02"},
        ]
        mock_ticker = _mock_ticker(news=fake_news)
        with patch("alpha_agent.tools.news.yf.Ticker", return_value=mock_ticker):
            result = search_news("AAPL", max_results=5)

        assert result["query"] == "AAPL"
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Apple beats earnings"

    def test_empty_query(self) -> None:
        result = search_news("")
        assert "error" in result

    def test_whitespace_query(self) -> None:
        result = search_news("   ")
        assert "error" in result

    def test_max_results_respected(self) -> None:
        fake_news = [{"title": f"Article {i}", "publisher": "X", "link": "", "providerPublishTime": ""} for i in range(10)]
        mock_ticker = _mock_ticker(news=fake_news)
        with patch("alpha_agent.tools.news.yf.Ticker", return_value=mock_ticker):
            result = search_news("AAPL", max_results=3)

        assert result["count"] == 3

    def test_no_news_returns_empty(self) -> None:
        mock_ticker = _mock_ticker(news=[])
        with patch("alpha_agent.tools.news.yf.Ticker", return_value=mock_ticker):
            result = search_news("ZZZZZZ")

        assert result["count"] == 0
        assert result["results"] == []

    def test_network_error(self) -> None:
        with patch("alpha_agent.tools.news.yf.Ticker", side_effect=ConnectionError("timeout")):
            result = search_news("AAPL")

        assert "error" in result


# ======================================================================
# Tests: get_company_info
# ======================================================================


class TestCompanyInfo:
    """get_company_info returns structured company data."""

    def test_returns_sector(self) -> None:
        fake_info = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2_800_000_000_000,
            "trailingPE": 29.5,
            "forwardPE": 25.0,
            "dividendYield": 0.005,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 164.08,
            "longBusinessSummary": "Apple Inc. designs, manufactures, and markets smartphones...",
        }
        mock_ticker = _mock_ticker(info=fake_info)
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_company_info("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["industry"] == "Consumer Electronics"
        assert result["market_cap"] == 2_800_000_000_000
        assert result["pe_ratio"] == 29.5
        assert result["forward_pe"] == 25.0

    def test_empty_info_returns_error(self) -> None:
        mock_ticker = _mock_ticker(info={})
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_company_info("INVALID")

        assert "error" in result

    def test_missing_fields_use_defaults(self) -> None:
        mock_ticker = _mock_ticker(info={"shortName": "Test Corp"})
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_company_info("TEST")

        assert result["name"] == "Test Corp"
        assert result["sector"] == "N/A"
        assert result["pe_ratio"] is None

    def test_description_truncated(self) -> None:
        long_summary = "A" * 1000
        mock_ticker = _mock_ticker(info={"shortName": "X", "longBusinessSummary": long_summary})
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_company_info("X")

        assert len(result["description"]) <= 500


# ======================================================================
# Tests: TOOL_REGISTRY
# ======================================================================


class TestToolRegistry:
    """The tool registry exposes all tools correctly."""

    def test_all_five_tools_registered(self) -> None:
        expected = {"get_stock_price", "get_stock_history", "calculate_risk_metrics", "search_news", "get_company_info"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_all_tools_have_fn(self) -> None:
        for name, tool in TOOL_REGISTRY.items():
            assert tool.fn is not None, f"Tool '{name}' has no callable"

    def test_all_tools_have_ollama_schema(self) -> None:
        for name, tool in TOOL_REGISTRY.items():
            schema = tool.to_ollama_tool()
            assert schema["type"] == "function"
            assert schema["function"]["name"] == name

    def test_price_tool_has_required_symbol(self) -> None:
        schema = TOOL_REGISTRY["get_stock_price"].to_ollama_tool()
        params = schema["function"]["parameters"]
        assert "symbol" in params["required"]

    def test_history_tool_has_optional_period(self) -> None:
        schema = TOOL_REGISTRY["get_stock_history"].to_ollama_tool()
        params = schema["function"]["parameters"]
        assert "period" in params["properties"]
        assert "period" not in params["required"]  # optional


# ======================================================================
# Tests: tool functions are directly callable
# ======================================================================


class TestToolDirectCalls:
    """Verify tool functions work when called directly (not via registry)."""

    def test_get_stock_price_returns_dict(self) -> None:
        mock_ticker = _mock_ticker(fast_info=_make_fast_info())
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_price("AAPL")
        assert isinstance(result, dict)
        assert "symbol" in result or "error" in result

    def test_get_stock_history_returns_dict(self) -> None:
        mock_ticker = _mock_ticker(history_df=_make_history_df())
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_stock_history("AAPL")
        assert isinstance(result, dict)

    def test_calculate_risk_metrics_returns_dict(self) -> None:
        mock_ticker = _mock_ticker(history_df=_make_history_df(n=63))
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = calculate_risk_metrics("AAPL")
        assert isinstance(result, dict)

    def test_search_news_returns_dict(self) -> None:
        mock_ticker = _mock_ticker(news=[])
        with patch("alpha_agent.tools.news.yf.Ticker", return_value=mock_ticker):
            result = search_news("AAPL")
        assert isinstance(result, dict)

    def test_get_company_info_returns_dict(self) -> None:
        mock_ticker = _mock_ticker(info={"shortName": "X"})
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            result = get_company_info("AAPL")
        assert isinstance(result, dict)


# ======================================================================
# Tests: integration with agent Tool schema
# ======================================================================


class TestToolIntegration:
    """Verify tools plug into the agent's Tool schema correctly."""

    def test_registry_tool_matches_ollama_format(self) -> None:
        """Every registered tool should produce a valid Ollama tool dict."""
        for tool in TOOL_REGISTRY.values():
            ollama = tool.to_ollama_tool()
            assert "function" in ollama
            assert "name" in ollama["function"]
            assert "parameters" in ollama["function"]
            assert "properties" in ollama["function"]["parameters"]

    def test_tool_fn_can_be_called_with_kwargs(self) -> None:
        """Agent._execute_tool calls tool.fn(**kwargs); verify this pattern works."""
        mock_ticker = _mock_ticker(fast_info=_make_fast_info())
        with patch("alpha_agent.tools.market.yf.Ticker", return_value=mock_ticker):
            tool = TOOL_REGISTRY["get_stock_price"]
            result = tool.fn(symbol="AAPL")
        assert isinstance(result, dict)
