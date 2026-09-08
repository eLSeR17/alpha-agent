"""Anti-hallucination guardrail.

Compares numeric values found in the agent's response against the raw
numbers returned by the tools.  Flags any number in the response that
does not appear in (or is significantly different from) the tool data.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Guardrail, ValidationResult

# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

# Matches integers, decimals, and numbers with commas/thousands separators.
# Also handles negative numbers and percentages.
_NUMBER_RE = re.compile(
    r"(?<![/\w])"           # not preceded by / or word char (avoids partial matches)
    r"[-+]?"                 # optional sign
    r"(?:\d{1,3}(?:,\d{3})+"  # e.g. 1,234,567
    r"|\d+)"                # plain digits
    r"(?:\.\d+)?"           # optional decimal
    r"(?:%)?"               # optional percent sign
)


def _parse_number(raw: str) -> float | None:
    """Try to parse a extracted number string into a float.

    Returns ``None`` if parsing fails (e.g. a year like "2025" that
    shouldn't be cross-checked is still parsed, but the caller decides
    whether to use it).
    """
    cleaned = raw.replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


class AntiHallucinationGuardrail(Guardrail):
    """Ensures the agent does not fabricate data.

    Parameters
    ----------
    tolerance:
        Relative tolerance for considering two numbers "matching".
        Default ``0.01`` means numbers within 1% of each other are OK.
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    # -- Guardrail interface -------------------------------------------------

    def validate_input(self, query: str) -> ValidationResult:
        """AntiHallucination does not inspect raw queries."""
        return ValidationResult(allowed=True)

    def validate_output(self, response: str) -> ValidationResult:
        """This is a thin wrapper — prefer :meth:`verify_grounding`."""
        # Default: no tool results to check against
        return ValidationResult(allowed=True)

    # -- Core verification ---------------------------------------------------

    def verify_grounding(
        self,
        response: str,
        tool_results: list[dict[str, Any]],
    ) -> ValidationResult:
        """Cross-check numbers in *response* against *tool_results*.

        Parameters
        ----------
        response:
            The agent's final answer text.
        tool_results:
            List of dicts (typically ``json.loads(tool_result.output)``)
            containing the raw tool data.

        Returns
        -------
        ValidationResult
            ``allowed=True`` if all response numbers are grounded,
            ``allowed=False`` with ``details["ungrounded"]`` listing
            the suspicious numbers.
        """
        response_numbers = self.extract_numbers(response)

        # Build a set of reference numbers from tool results
        tool_numbers = self._collect_tool_numbers(tool_results)

        # Skip trivial numbers (< 1) and years (1900-2100) from checks
        # — these are usually dates, sentence counts, etc.
        significant = [
            n for n in response_numbers if abs(n) >= 1.0 and not (1900 <= n <= 2100)
        ]

        ungrounded: list[dict[str, Any]] = []

        for num in significant:
            if not self._is_grounded(num, tool_numbers):
                ungrounded.append({
                    "number": num,
                    "nearest_tool_number": self._nearest(num, tool_numbers),
                })

        if ungrounded:
            return ValidationResult(
                allowed=False,
                reason=(
                    f"{len(ungrounded)} number(s) in the response could not "
                    f"be verified against tool results"
                ),
                details={
                    "ungrounded": ungrounded,
                    "response_number_count": len(response_numbers),
                    "tool_number_count": len(tool_numbers),
                },
            )

        return ValidationResult(
            allowed=True,
            details={
                "response_numbers_checked": len(significant),
                "tool_numbers_available": len(tool_numbers),
            },
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        """Extract all numeric values from *text*.

        Returns a list of floats (percentages are kept as-is, e.g. ``2.5``
        for ``2.5%``).
        """
        matches = _NUMBER_RE.findall(text)
        numbers: list[float] = []
        for m in matches:
            parsed = _parse_number(m)
            if parsed is not None:
                numbers.append(parsed)
        return numbers

    @staticmethod
    def _collect_tool_numbers(tool_results: list[dict[str, Any]]) -> list[float]:
        """Recursively extract all numeric values from tool-result dicts."""
        numbers: list[float] = []

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item)
            elif isinstance(obj, (int, float)):
                numbers.append(float(obj))
            elif isinstance(obj, str):
                parsed = _parse_number(obj)
                if parsed is not None:
                    numbers.append(parsed)

        for tr in tool_results:
            _walk(tr)
        return numbers

    def _is_grounded(self, value: float, references: list[float]) -> bool:
        """Check if *value* is within tolerance of any reference number."""
        for ref in references:
            if ref == 0.0:
                if abs(value) < 1e-9:
                    return True
                continue
            relative_diff = abs(value - ref) / abs(ref)
            if relative_diff <= self.tolerance:
                return True
        return False

    def _nearest(self, value: float, references: list[float]) -> float | None:
        """Return the closest reference number, or ``None`` if empty."""
        if not references:
            return None
        return min(references, key=lambda r: abs(r - value))
