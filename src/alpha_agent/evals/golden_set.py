"""Golden dataset loading and validation.

The golden set is a curated collection of evaluation cases (queries plus
annotated expectations) used to assess the quality of AlphaAgent in a
reproducible way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvalCase(BaseModel):
    """A single evaluation case in the golden dataset.

    Attributes
    ----------
    query:
        The user query to run against the agent.
    category:
        A free-form tag grouping related cases (e.g. ``"price"``,
        ``"multiple-tools"``, ``"failure-mode"``, ``"edge-case"``,
        ``"malicious"``).
    expected_tool:
        The tool the agent is expected to invoke.  Use the special value
        ``"multiple"`` when several tools are expected, or ``None`` when no
        tool should be called at all (e.g. a blocked malicious query or a
        clarification request whose correct behaviour is to ask the user
        rather than call any tool).
    expected_symbol:
        The ticker symbol the agent is expected to resolve, if any.
    min_evidence:
        A list of keys/terms the answer or tool output must contain for the
        case to be considered grounded.  An empty list disables the check.
    should_block:
        ``True`` when this case represents a malicious query that the
        guardrails must reject before any tool call happens.
    notes:
        Optional human-readable intent for the case.
    """

    query: str
    category: str
    expected_tool: str | None = None
    expected_symbol: str | None = None
    min_evidence: list[str] = Field(default_factory=list)
    should_block: bool = False
    notes: str = ""

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @property
    def is_malicious(self) -> bool:
        """Return ``True`` when the guardrails are expected to block the query."""
        return self.should_block and self.category == "malicious"


def load_golden_set(path: str | Path) -> list[EvalCase]:
    """Load and validate a golden set from a JSON file.

    Parameters
    ----------
    path:
        Path to the ``golden_set.json`` file.

    Returns
    -------
    list[EvalCase]
        The validated evaluation cases.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON payload is not a list of valid case objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Golden set must be a JSON list, got {type(raw).__name__}")

    if not raw:
        raise ValueError("Golden set must contain at least one case")

    # model_validate_list reports all validation errors in one go.
    return [EvalCase.model_validate(item) for item in raw]
