"""Base classes for the guardrail system.

Every guardrail implements a common interface so the :class:`GuardedAlphaAgent`
can chain them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a guardrail check.

    Attributes
    ----------
    allowed:
        ``True`` if the input/output passes the guardrail.
    reason:
        Human-readable explanation (empty string when allowed).
    details:
        Arbitrary extra data the guardrail wants to attach (e.g. which
        pattern matched, which numbers were flagged, etc.).
    """

    allowed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guardrail interface
# ---------------------------------------------------------------------------


class Guardrail(ABC):
    """Abstract base for all guardrails.

    Subclasses override ``validate_input`` and/or ``validate_output``.
    The default implementations are permissive pass-throughs.
    """

    @abstractmethod
    def validate_input(self, query: str) -> ValidationResult:
        """Check whether the user query is safe to process.

        Parameters
        ----------
        query:
            The raw user query string.

        Returns
        -------
        ValidationResult
            ``allowed=True`` to proceed, ``allowed=False`` to reject.
        """

    @abstractmethod
    def validate_output(self, response: str) -> ValidationResult:
        """Check whether the agent's response is safe to return.

        Parameters
        ----------
        response:
            The final answer text produced by the agent.

        Returns
        -------
        ValidationResult
            ``allowed=True`` to return as-is, ``allowed=False`` to flag
            (the caller decides how to handle rejections).
        """
