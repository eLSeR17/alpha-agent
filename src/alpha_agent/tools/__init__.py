"""Financial tools for the AlphaAgent agent.

Each tool is a pure Python function that returns a JSON-serialisable dict.
The ``TOOL_REGISTRY`` exposes all tools as :class:`Tool` instances ready for
Ollama function-calling.
"""

from __future__ import annotations

from ..schemas import Tool, ToolParameter
from .market import get_company_info, get_stock_history, get_stock_price
from .news import search_news
from .risk import calculate_risk_metrics

# ---------------------------------------------------------------------------
# Registry — single source of truth for all available tools
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Tool] = {
    "get_stock_price": Tool(
        name="get_stock_price",
        description="Get the current stock price, daily change, and currency for a given ticker symbol.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Ticker symbol, e.g. AAPL, MSFT, TSLA", required=True),
        ],
        fn=get_stock_price,
    ),
    "get_stock_history": Tool(
        name="get_stock_history",
        description=(
            "Get historical price data (high, low, avg close) for a ticker over "
            "a specified time period. Use this tool when the user asks about a "
            "stock's price or performance over ANY time range — e.g. 'last month', "
            "'past week', 'year-to-date', 'last 3 months', '1-year history', "
            "'past year', 'historical', or any other period reference."
        ),
        parameters=[
            ToolParameter(name="symbol", type="string", description="Ticker symbol, e.g. AAPL", required=True),
            ToolParameter(
                name="period",
                type="string",
                description="Lookback period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
                required=False,
                enum=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            ),
        ],
        fn=get_stock_history,
    ),
    "calculate_risk_metrics": Tool(
        name="calculate_risk_metrics",
        description="Calculate annualised volatility, Sharpe ratio, and maximum drawdown for a ticker.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Ticker symbol, e.g. AAPL", required=True),
            ToolParameter(
                name="period",
                type="string",
                description="Lookback period for risk calculation (default: 3mo)",
                required=False,
                enum=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            ),
        ],
        fn=calculate_risk_metrics,
    ),
    "search_news": Tool(
        name="search_news",
        description=(
            "Search recent financial news headlines and summaries for a query "
            "(e.g. ticker or keyword). Use ONLY when the user explicitly asks "
            "for news, articles, headlines, or recent events. Do NOT use as a "
            "fallback for price or history queries."
        ),
        parameters=[
            ToolParameter(name="query", type="string", description="Search query, e.g. AAPL earnings, Fed rate hike", required=True),
            ToolParameter(name="max_results", type="number", description="Max results to return (default 5)", required=False),
        ],
        fn=search_news,
    ),
    "get_company_info": Tool(
        name="get_company_info",
        description="Get company fundamentals: name, sector, industry, market cap, P/E ratio, and description.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Ticker symbol, e.g. AAPL", required=True),
        ],
        fn=get_company_info,
    ),
}

__all__ = [
    "TOOL_REGISTRY",
    "get_stock_price",
    "get_stock_history",
    "calculate_risk_metrics",
    "search_news",
    "get_company_info",
]
