"""Tool-protection guardrail.

Validates ticker symbols before they reach the financial tools, and
enforces per-tool rate limiting with a sliding window.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

from .base import Guardrail, ValidationResult

# ---------------------------------------------------------------------------
# Ticker symbol validation
# ---------------------------------------------------------------------------

# Valid formats: AAPL, MSFT, ^GSPC (S&P 500), BTC-USD, BRK-B, etc.
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^]{1,10}$")


class ToolGuardrail(Guardrail):
    """Protects financial tools from malicious or erroneous use.

    Parameters
    ----------
    calls_per_minute:
        Maximum calls allowed per tool within a 60-second sliding window.
    """

    def __init__(self, calls_per_minute: int = 10) -> None:
        self.calls_per_minute = calls_per_minute
        self._call_log: dict[str, list[float]] = defaultdict(list)

    # -- Guardrail interface -------------------------------------------------

    def validate_input(self, query: str) -> ValidationResult:
        """ToolGuardrail does not inspect free-text queries."""
        return ValidationResult(allowed=True)

    def validate_output(self, response: str) -> ValidationResult:
        """ToolGuardrail does not inspect free-text responses."""
        return ValidationResult(allowed=True)

    # -- Symbol validation ---------------------------------------------------

    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """Return ``True`` if *symbol* matches the allowed ticker format."""
        return bool(_SYMBOL_RE.match(symbol.upper()))

    def validate_symbol(self, symbol: str) -> ValidationResult:
        """Validate a ticker symbol string.

        Returns a :class:`ValidationResult` with ``allowed=True`` for valid
        tickers and ``allowed=False`` with an explanatory reason otherwise.
        """
        if not symbol or not symbol.strip():
            return ValidationResult(
                allowed=False,
                reason="Symbol is empty",
                details={"symbol": symbol},
            )

        if not self.is_valid_symbol(symbol):
            return ValidationResult(
                allowed=False,
                reason=f"Invalid ticker format: {symbol!r}",
                details={"symbol": symbol, "expected_format": "^[A-Z0-9.\\-^]{1,10}$"},
            )

        return ValidationResult(allowed=True, details={"symbol": symbol.upper()})

    # -- Rate limiting -------------------------------------------------------

    def check_rate_limit(self, tool_name: str) -> ValidationResult:
        """Check whether *tool_name* has exceeded its call budget.

        Uses a sliding window of 60 seconds.  Old timestamps are pruned
        on each call so the window naturally resets.
        """
        now = time.monotonic()
        window = 60.0

        # Prune timestamps outside the window
        self._call_log[tool_name] = [
            t for t in self._call_log[tool_name] if now - t < window
        ]

        if len(self._call_log[tool_name]) >= self.calls_per_minute:
            oldest = self._call_log[tool_name][0]
            retry_after = window - (now - oldest)
            return ValidationResult(
                allowed=False,
                reason=(
                    f"Rate limit exceeded for '{tool_name}': "
                    f"{len(self._call_log[tool_name])}/{self.calls_per_minute} "
                    f"in the last 60s"
                ),
                details={
                    "tool": tool_name,
                    "calls_in_window": len(self._call_log[tool_name]),
                    "limit": self.calls_per_minute,
                    "retry_after_seconds": round(retry_after, 1),
                },
            )

        # Record this call
        self._call_log[tool_name].append(now)
        return ValidationResult(
            allowed=True,
            details={
                "tool": tool_name,
                "calls_in_window": len(self._call_log[tool_name]),
                "limit": self.calls_per_minute,
            },
        )

    # -- Reset (useful for tests) -------------------------------------------

    def reset(self) -> None:
        """Clear all rate-limit counters."""
        self._call_log.clear()
