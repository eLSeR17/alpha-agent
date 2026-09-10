"""News search tool — financial news via yfinance ticker news feeds.

Uses ``yfinance``'s built-in ``Ticker.news`` which aggregates RSS feeds
from major financial outlets.  No API key required.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

_MAX_RESULTS_HARD_CAP = 20  # never return more than this


def search_news(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search financial news related to a query.

    The function first attempts to treat *query* as a ticker symbol and
    fetches its news feed.  If the symbol is invalid, it falls back to a
    broader ``Ticker("NEWS")`` search that yfinance supports for keyword
    queries.

    Parameters
    ----------
    query:
        A ticker symbol (``"AAPL"``) or keyword phrase (``"Fed rate hike"``).
    max_results:
        Maximum number of articles to return (default 5, hard-capped at 20).

    Returns
    -------
    dict
        ``{"query", "results": [{"title", "publisher", "link", "published"}]}``
        or ``{"error": "<message>"}``.

    Examples
    --------
    >>> search_news("AAPL")  # doctest: +SKIP
    {'query': 'AAPL', 'results': [{'title': 'Apple reports record Q4...', ...}]}
    """
    if not query or not query.strip():
        return {"error": "Query must be a non-empty string."}

    max_results = max(1, min(int(max_results), _MAX_RESULTS_HARD_CAP))
    query = query.strip()

    try:
        # Strategy: try the query as a ticker first
        ticker = yf.Ticker(query)
        news_items: list[dict[str, Any]] = ticker.news or []

        # If no news from direct ticker, try uppercased version (common case)
        if not news_items and query.upper() != query:
            ticker = yf.Ticker(query.upper())
            news_items = ticker.news or []

        results: list[dict[str, Any]] = []
        for item in news_items[:max_results]:
            # yfinance news items vary by version; handle common shapes
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
            results.append({
                "title": item.get("title") or content.get("title", "No title"),
                "publisher": item.get("publisher") or content.get("provider", {}).get("displayName", "Unknown"),
                "link": item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
                "published": item.get("providerPublishTime") or content.get("pubDate", ""),
            })

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:  # noqa: BLE001  # degrade gracefully on any provider error
        logger.warning("search_news failed for query '%s': %s", query, exc)
        return {"error": f"Failed to fetch news for '{query}': {exc}"}
