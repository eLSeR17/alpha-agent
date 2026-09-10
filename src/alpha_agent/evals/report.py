"""Eval report -- aggregate results into summary metrics and persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .runner import EvalResult


class EvalReport(BaseModel):
    """Summary report of an evaluation run.

    Attributes
    ----------
    total:
        Total number of cases evaluated.
    passed:
        Number of cases that passed.
    failed:
        Number of cases that failed.
    pass_rate:
        Fraction of cases passed (0.0 - 1.0).
    mean_score:
        Mean weighted score across all non-blocked cases.
    criteria_breakdown:
        Mean score per criterion (0.0 - 1.0).
    category_breakdown:
        Per-category pass rate (0.0 - 1.0).
    results:
        The individual :class:`EvalResult` objects.
    generated_at:
        ISO timestamp of when the report was produced.
    """

    total: int
    passed: int
    failed: int
    pass_rate: float
    mean_score: float
    criteria_breakdown: dict[str, float] = Field(default_factory=dict)
    category_breakdown: dict[str, float] = Field(default_factory=dict)
    results: list[EvalResult] = Field(default_factory=list)
    generated_at: str = ""

    @classmethod
    def from_results(cls, results: list[EvalResult]) -> EvalReport:
        """Build a report by aggregating *results*."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Mean score across cases that were actually scored by the judge.
        # Blocked (malicious) cases carry a neutral criteria; exclude them so
        # the mean stays representative of genuine answer quality.
        scored = [r.score for r in results if not r.case.is_malicious]
        mean_score = round(sum(scored) / len(scored), 4) if scored else 0.0

        criteria_names = ["grounded", "correct_tool", "clarity", "relevance"]
        criteria_breakdown: dict[str, float] = {}
        for name in criteria_names:
            values = [getattr(r.criteria, name) for r in results if not r.case.is_malicious]
            criteria_breakdown[name] = round(sum(values) / len(values), 4) if values else 0.0

        # Per-category pass rate.
        by_category: dict[str, list[bool]] = {}
        for r in results:
            by_category.setdefault(r.case.category, []).append(r.passed)
        category_breakdown = {
            cat: round(sum(oks) / len(oks), 4)
            for cat, oks in by_category.items()
        }

        return cls(
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=round(passed / total, 4) if total else 0.0,
            mean_score=mean_score,
            criteria_breakdown=criteria_breakdown,
            category_breakdown=category_breakdown,
            results=results,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


def print_report(report: EvalReport) -> None:
    """Print a human-readable report to stdout."""
    line = "=" * 60
    print(line)
    print("AlphaAgent Eval Report")
    print(line)
    print(f"Generated : {report.generated_at}")
    print(f"Total     : {report.total}")
    print(f"Passed    : {report.passed}")
    print(f"Failed    : {report.failed}")
    print(f"Pass rate : {report.pass_rate:.1%}")
    print(f"Mean score: {report.mean_score:.3f}")
    print(line)
    print("Criteria breakdown (mean 0-1)")
    for name, value in report.criteria_breakdown.items():
        print(f"  {name:<14}: {value:.3f}")
    print(line)
    print("Category breakdown (pass rate)")
    for name, value in report.category_breakdown.items():
        print(f"  {name:<18}: {value:.1%}")
    print(line)
    for ix, result in enumerate(report.results, start=1):
        status = "PASS" if result.passed else "FAIL"
        print(f"[{ix:>2}] {status}  {result.case.category:<16} "
              f"score={result.score:.3f}  {result.case.query[:60]}")
        for detail in result.details:
            print(f"       - {detail}")
    print(line)


def save_report(report: EvalReport, path: str | Path) -> Path:
    """Persist *report* as JSON to *path*.

    Returns
    -------
    Path
        The path the report was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_to_dict(report)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Convert a report to a JSON-serialisable dict."""
    return {
        "generated_at": report.generated_at,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "mean_score": report.mean_score,
        "criteria_breakdown": report.criteria_breakdown,
        "category_breakdown": report.category_breakdown,
        "results": [
            {
                "query": r.case.query,
                "category": r.case.category,
                "expected_tool": r.case.expected_tool,
                "expected_symbol": r.case.expected_symbol,
                "called_tools": r.called_tools,
                "used_expected_tool": r.used_expected_tool,
                "used_expected_symbol": r.used_expected_symbol,
                "was_blocked": r.was_blocked,
                "criteria": r.criteria.to_dict(),
                "score": r.score,
                "passed": r.passed,
                "details": r.details,
                "final_answer": r.final_answer,
            }
            for r in report.results
        ],
    }
