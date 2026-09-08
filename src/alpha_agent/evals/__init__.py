"""Evals for AlphaAgent -- golden dataset + LLM-as-judge.

This package provides a reproducible evaluation harness that measures the
quality of the AlphaAgent across a golden set of curated scenarios.  It
covers:

* a golden dataset loader (:mod:`golden_set`)
* a runner that executes the agent over a golden set (:mod:`runner`)
* an LLM-as-judge scoring responses against quality criteria (:mod:`judge`)
* a report builder that aggregates results into actionable metrics
  (:mod:`report`)
"""

from .golden_set import EvalCase, load_golden_set
from .judge import (
    BaseJudge,
    JudgeContext,
    JudgeCriteria,
    LLMJudge,
    MockJudge,
    DEFAULT_WEIGHTS,
)
from .report import EvalReport, print_report, save_report
from .runner import EvalResult, run_evals

__all__ = [
    "BaseJudge",
    "DEFAULT_WEIGHTS",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "JudgeContext",
    "JudgeCriteria",
    "LLMJudge",
    "MockJudge",
    "load_golden_set",
    "print_report",
    "run_evals",
    "save_report",
]
