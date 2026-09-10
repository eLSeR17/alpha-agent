#!/usr/bin/env python3
"""Run the AlphaAgent eval suite end-to-end.

Executes the agent over the golden dataset and produces an eval report with
an LLM-as-judge (Ollama) by default, or a deterministic mock judge with
``--mock``.

Usage
-----
    python3 scripts/run_evals.py                 # real Ollama judge
    python3 scripts/run_evals.py --mock          # deterministic, offline
    python3 scripts/run_evals.py --mock --json data/evals/custom.json

Exit codes
----------
    0  if the mean score meets the pass threshold
    1  if it does not (or on error)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``alpha_agent`` (root src) and ``alpha_agent.evals`` (staging src)
# importable when this script is run directly from the repo.  The script may
# live at ``scripts/`` (after promotion) or ``staging/scripts/`` (during
# development), so the repo root is found by walking up to the directory that
# holds ``src/alpha_agent`` and ``.git``.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR
while not (
    (_PROJECT_ROOT / "src" / "alpha_agent" / "__init__.py").exists()
    and (_PROJECT_ROOT / ".git").exists()
):
    _parent = _PROJECT_ROOT.parent
    if _parent == _PROJECT_ROOT:
        raise RuntimeError("Could not locate alpha-agent repo root from " + str(_SCRIPT_DIR))
    _PROJECT_ROOT = _parent

_ROOT_SRC = _PROJECT_ROOT / "src"
_STAGING_SRC = _PROJECT_ROOT / "staging" / "src"
# Put root src FIRST so the real ``alpha_agent`` package (with __init__.py)
# wins over the staging namespace portion that only holds the ``evals``
# subpackage.
if str(_STAGING_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGING_SRC))
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

import alpha_agent

# Make ``alpha_agent.evals`` (staging src) reachable from the imported package.
_staging_pkg = str(_STAGING_SRC / "alpha_agent")
if _staging_pkg not in list(alpha_agent.__path__):
    alpha_agent.__path__.append(_staging_pkg)

from alpha_agent import TOOL_REGISTRY
from alpha_agent.agent import AlphaAgent
from alpha_agent.evals import (
    EvalReport,
    LLMJudge,
    MockJudge,
    load_golden_set,
    print_report,
    run_evals,
    save_report,
)
from alpha_agent.evals.runner import DEFAULT_PASS_THRESHOLD
from alpha_agent.guarded_agent import GuardedAlphaAgent
from alpha_agent.llm import OllamaClient
from alpha_agent.schemas import LLMResponse


class _StubLLM:
    """Offline stand-in for the agent's LLM.

    Returns a plain final answer with no tool calls, so ``--mock`` runs the
    full harness (load -> guard -> run -> judge -> report) without Ollama or
    network access.  Tool-call metadata and symbol checks will therefore fail
    on most cases -- that is expected: ``--mock`` validates the harness
    plumbing, not answer quality.
    """

    def __init__(self) -> None:
        self.model = "stub"
        self.base_url = "offline"

    def chat_with_tools(self, messages, tools, temperature: float = 0.3):
        return LLMResponse(
            content="Based on the available information, here is an answer.",
            tool_calls=[],
            done=True,
            raw={},
        )

    def close(self) -> None:
        return None


def _resolve_default_paths(root: Path, args) -> tuple[Path, Path]:
    """Resolve golden-set input and report output paths.

    During staging the data lives under ``staging/data``; after promotion it
    lives directly under ``data``.  We search both so the same script works in
    either layout.
    """
    candidates = [
        root / "staging" / "data" / "golden" / "golden_set.json",
        root / "data" / "golden" / "golden_set.json",
    ]
    golden = Path(args.golden) if args.golden else next((c for c in candidates if c.exists()), candidates[0])

    if args.report_dir:
        report_dir = Path(args.report_dir)
    else:
        # Write under the same subtree the golden set was found in.
        report_dir = golden.parent.parent / "evals" if golden.parent.name == "golden" else root / "data" / "evals"
    return golden, report_dir


def main(argv: list[str] | None = None) -> int:
    """Execute the evals and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use deterministic mock judge (offline)")
    parser.add_argument("--golden", type=str, default=None, help="path to golden_set.json")
    parser.add_argument("--report-dir", type=str, default=None, help="output directory for reports")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PASS_THRESHOLD, help="pass threshold (default 0.7)")
    parser.add_argument("--model", type=str, default=None, help="Ollama model for the LLM judge")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    # Keep the default at WARNING unless --verbose is passed.
    if not args.verbose:
        logging.basicConfig(level=logging.WARNING)

    root = _PROJECT_ROOT
    golden_path, report_dir = _resolve_default_paths(root, args)

    if not golden_path.exists():
        print(f"ERROR: golden set not found at {golden_path}", file=sys.stderr)
        return 1

    cases = load_golden_set(golden_path)
    print(f"Loaded {len(cases)} golden cases from {golden_path}")

    # Build the guarded agent with the real financial tools.
    tools = list(TOOL_REGISTRY.values())

    if args.mock:
        # Fully offline: stub LLM for the agent + deterministic mock judge.
        llm = _StubLLM()
        judge = MockJudge()
        print("Using deterministic MOCK judge + stub agent LLM (no Ollama / network)")
    else:
        llm = OllamaClient()
        model = args.model
        judge = LLMJudge(model=model) if model else LLMJudge()
        print(f"Using LLM judge: {judge.model} @ {judge.base_url}")
        print(f"Agent LLM: {llm.model} @ {llm.base_url}")

    base_agent = AlphaAgent(llm_client=llm, tools=tools)
    agent = GuardedAlphaAgent(base_agent)

    results = run_evals(agent, cases, judge, pass_threshold=args.threshold)
    report = EvalReport.from_results(results)
    print_report(report)

    timestamp = report.generated_at.replace(":", "").replace("+", "_")
    report_path = report_dir / f"report_{timestamp}.json"
    save_report(report, report_path)
    print(f"\nReport saved to {report_path}")

    # Exit code semantics: 0 if mean score meets threshold, else 1.
    if report.mean_score >= args.threshold:
        print(f"\nPASS: mean score {report.mean_score:.3f} >= threshold {args.threshold:.2f}")
        return 0
    print(f"\nFAIL: mean score {report.mean_score:.3f} < threshold {args.threshold:.2f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
