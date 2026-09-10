"""Market data tools — stock price, history, and company info.

All data comes from ``yfinance`` (free, no API key required).
Network errors and invalid symbols are caught and returned as structured dicts
so the agent can reason about failures.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Valid periods accepted by yfinance History API
_VALID_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})


def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get the current stock price, daily change percentage, and currency.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``, ``"MSFT"``).

    Returns
    -------
    dict
        ``{"symbol", "price", "currency", "change_pct", "previous_close"}``
        or ``{"error": "<message>"}`` on failure.

    Examples
    --------
    >>> get_stock_price("AAPL")  # doctest: +SKIP
    {'symbol': 'AAPL', 'price': 185.50, 'currency': 'USD', 'change_pct': 1.23, 'previous_close': 183.27}
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        # fast_info is lighter than full .info; still has price fields
        price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        currency = getattr(info, "currency", "USD")

        if price is None:
            return {"error": f"No price data available for '{symbol}'. The symbol may be delisted or invalid."}

        change_pct = None
        if prev_close and prev_close != 0:
            change_pct = round(((price - prev_close) / prev_close) * 100, 4)

        return {
            "symbol": symbol.upper(),
            "price": round(float(price), 4),
            "currency": str(currency),
            "change_pct": change_pct,
            "previous_close": round(float(prev_close), 4) if prev_close else None,
        }
    except Exception as exc:  # noqa: BLE001  # degrade gracefully on any provider error
        logger.warning("get_stock_price failed for %s: %s", symbol, exc)
        return {"error": f"Failed to fetch price for '{symbol}': {exc}"}


def get_stock_history(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """Get historical price summary (high, low, avg close) over a period.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    period:
        Lookback period — one of ``1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y,
        10y, ytd, max``.  Defaults to ``"1mo"``.

    Returns
    -------
    dict
        ``{"symbol", "period", "data_points", "high", "low", "avg", "start_date", "end_date"}``
        or ``{"error": "<message>"}``.

    Examples
    --------
    >>> get_stock_history("AAPL", period="3mo")  # doctest: +SKIP
    {'symbol': 'AAPL', 'period': '3mo', 'data_points': 63, 'high': 198.23, 'low': 164.08, ...}
    """
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Valid options: {sorted(_VALID_PERIODS)}"}

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return {"error": f"No historical data for '{symbol}' with period '{period}'."}

        closes = hist["Close"]
        return {
            "symbol": symbol.upper(),
            "period": period,
            "data_points": len(closes),
            "high": round(float(hist["High"].max()), 4),
            "low": round(float(hist["Low"].min()), 4),
            "avg": round(float(closes.mean()), 4),
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
        }
    except Exception as exc:  # noqa: BLE001  # degrade gracefully on any provider error
        logger.warning("get_stock_history failed for %s: %s", symbol, exc)
        return {"error": f"Failed to fetch history for '{symbol}': {exc}"}


def get_company_info(symbol: str) -> dict[str, Any]:
    """Get company fundamentals: name, sector, industry, market cap, P/E.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    dict
        ``{"symbol", "name", "sector", "industry", "market_cap",
        "pe_ratio", "dividend_yield", "fifty_two_week_high",
        "fifty_two_week_low"}`` or ``{"error": "<message>"}``.

    Examples
    --------
    >>> get_company_info("AAPL")  # doctest: +SKIP
    {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', ...}
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or info.get("trailingPegRatio") is None and info.get("shortName") is None:
            return {"error": f"No company info available for '{symbol}'."}

        return {
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "description": (info.get("longBusinessSummary") or "")[:500],
        }
    except Exception as exc:  # noqa: BLE001  # degrade gracefully on any provider error
        logger.warning("get_company_info failed for %s: %s", symbol, exc)
        return {"error": f"Failed to fetch company info for '{symbol}': {exc}"}
