"""Risk metrics tool — volatility, Sharpe ratio, max drawdown.

All calculations use ``yfinance`` price history and standard quantitative
finance formulas.  No external API keys required.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Risk-free rate assumption for Sharpe ratio (approximate annualised US T-bill)
_RISK_FREE_RATE = 0.045  # ~4.5 % as of 2024-2025

_VALID_PERIODS = frozenset({"1mo", "3mo", "6mo", "1y", "2y", "5y"})


def calculate_risk_metrics(symbol: str, period: str = "3mo") -> dict[str, Any]:
    """Calculate annualised volatility, Sharpe ratio, and max drawdown.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    period:
        Lookback period for the calculation.  One of ``1mo, 3mo, 6mo, 1y,
        2y, 5y``.  Defaults to ``"3mo"``.

    Returns
    -------
    dict
        ``{"symbol", "period", "volatility", "sharpe_ratio", "max_drawdown",
        "annualised_return", "data_points"}`` or ``{"error": "<message>"}``.

    Notes
    -----
    * **Volatility** = annualised standard deviation of daily log returns.
    * **Sharpe ratio** = (annualised return − risk-free rate) / volatility.
    * **Max drawdown** = worst peak-to-trough decline (negative percentage).

    Examples
    --------
    >>> calculate_risk_metrics("AAPL", period="3mo")  # doctest: +SKIP
    {'symbol': 'AAPL', 'volatility': 0.25, 'sharpe_ratio': 1.2, 'max_drawdown': -0.15, ...}
    """
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Valid options: {sorted(_VALID_PERIODS)}"}

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty or len(hist) < 2:
            return {"error": f"Insufficient data for '{symbol}' with period '{period}' (need >=2 data points)."}

        closes = hist["Close"].astype(float)
        data_points = len(closes)

        # --- Daily log returns (vectorised; matplotlib-agnostic) ---
        log_returns = np.log(closes / closes.shift(1)).dropna()

        if len(log_returns) < 2:
            return {"error": f"Not enough return data for '{symbol}' to compute risk metrics."}

        # --- Volatility (annualised) ---
        # Trading days ≈ 252
        daily_vol = float(log_returns.std())
        annualised_vol = daily_vol * math.sqrt(252)

        # --- Annualised return ---
        total_return = float(closes.iloc[-1] / closes.iloc[0] - 1)
        # Scale to annual based on actual trading days observed
        annualised_return = total_return * (252 / data_points)

        # --- Sharpe ratio ---
        sharpe = (annualised_return - _RISK_FREE_RATE) / annualised_vol if annualised_vol != 0 else 0.0

        # --- Max drawdown ---
        running_max = closes.cummax()
        drawdowns = (closes - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        return {
            "symbol": symbol.upper(),
            "period": period,
            "volatility": round(annualised_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "annualised_return": round(annualised_return, 4),
            "data_points": data_points,
            "risk_free_rate": _RISK_FREE_RATE,
        }
    except Exception as exc:  # noqa: BLE001  # degrade gracefully on any provider error
        logger.warning("calculate_risk_metrics failed for %s: %s", symbol, exc)
        return {"error": f"Failed to calculate risk metrics for '{symbol}': {exc}"}
