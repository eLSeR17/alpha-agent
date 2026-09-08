"""Financial-domain guardrail.

Blocks prompt-injection attacks, illegal-activity requests, and
impersonation attempts on input.  Ensures a financial disclaimer is
present in the output.
"""

from __future__ import annotations

import re

from .base import Guardrail, ValidationResult

# ---------------------------------------------------------------------------
# Input patterns to block
# ---------------------------------------------------------------------------

# Prompt injection – classic variants
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(are|were)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|rules|constraints)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you|your)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(system|your)\s+(prompt|instructions?)", re.IGNORECASE),
    re.compile(r"from\s+now\s+on\s+you\s+(will|must|should)\s+(behave|act|respond)", re.IGNORECASE),
]

# Illegal / unethical activity
_ILLEGAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(insider\s+trading|market\s+manipulation|pump\s+and\s+dump)\b", re.IGNORECASE),
    re.compile(r"\b(money\s+laundering|tax\s+evasion|fraud)\b", re.IGNORECASE),
    re.compile(r"\b(hack|exploit|bypass\s+security)\b", re.IGNORECASE),
]

# Impersonation / role hijack
_IMPERSONATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"you\s+are\s+(a|an)\s+(unrestricted|unfiltered|jailbroken)\s+(ai|model|assistant)", re.IGNORECASE),
    re.compile(r"(pretend|act)\s+(you\s+)?(have\s+no\s+restrictions|are\s+unfiltered)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Financial disclaimer template
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "\n\n---\n*Disclaimer: This information is for educational and informational "
    "purposes only and does not constitute financial advice. Always consult a "
    "qualified financial advisor before making investment decisions.*"
)

# Regex that matches an existing disclaimer so we don't duplicate it.
_DISCLAIMER_RE = re.compile(
    r"(?:disclaimer|for\s+informational\s+purposes\s+only|does\s+not\s+constitute\s+financial\s+advice)",
    re.IGNORECASE,
)


class FinancialGuardrail(Guardrail):
    """Protects the financial domain of the agent.

    *Input* – rejects prompt injection, illegal-activity requests, and
    impersonation attempts.

    *Output* – ensures a financial disclaimer is present.
    """

    # -- input ---------------------------------------------------------------

    def validate_input(self, query: str) -> ValidationResult:  # noqa: D401
        """Reject queries that try to manipulate the agent."""
        for pat in _INJECTION_PATTERNS:
            m = pat.search(query)
            if m:
                return ValidationResult(
                    allowed=False,
                    reason="Prompt injection detected",
                    details={"pattern": pat.pattern, "match": m.group()},
                )

        for pat in _ILLEGAL_PATTERNS:
            m = pat.search(query)
            if m:
                return ValidationResult(
                    allowed=False,
                    reason="Request involves illegal or unethical activity",
                    details={"pattern": pat.pattern, "match": m.group()},
                )

        for pat in _IMPERSONATION_PATTERNS:
            m = pat.search(query)
            if m:
                return ValidationResult(
                    allowed=False,
                    reason="Impersonation / role-hijack attempt detected",
                    details={"pattern": pat.pattern, "match": m.group()},
                )

        return ValidationResult(allowed=True)

    # -- output --------------------------------------------------------------

    def validate_output(self, response: str) -> ValidationResult:
        """Ensure the response contains a financial disclaimer.

        If the disclaimer is missing, ``enrich`` should be called to append
        it.  This method only *validates* — it never mutates.
        """
        if _DISCLAIMER_RE.search(response):
            return ValidationResult(allowed=True)

        return ValidationResult(
            allowed=False,
            reason="Financial disclaimer missing from response",
            details={"disclaimer_text": DISCLAIMER},
        )

    # -- helper --------------------------------------------------------------

    @staticmethod
    def enrich(response: str) -> str:
        """Append the standard financial disclaimer if not already present."""
        if _DISCLAIMER_RE.search(response):
            return response
        return response + DISCLAIMER
