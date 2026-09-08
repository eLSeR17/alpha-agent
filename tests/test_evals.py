"""Unit tests for the eval harness -- no live Ollama calls.

These tests validate the golden-set loader, the runner, the deterministic
mock judge, and the report aggregator using a scripted fake agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alpha_agent.evals import (
    EvalReport,
    JudgeContext,
    JudgeCriteria,
    LLMJudge,
    MockJudge,
    load_golden_set,
    run_evals,
    save_report,
)
from alpha_agent.evals.golden_set import EvalCase

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """A scripted stand-in for the real agent.

    Returns a pre-built ``AgentResponse`` for every query, so the runner can
    be tested without a live LLM or financial tools.
    """

    def __init__(self, response) -> None:
        self.response = response

    def analyze(self, query: str):
        return self.response


def _agent_response(
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None = None,
    blocked: bool = False,
) -> Any:
    """Build a minimal ``AgentResponse``-like object."""
    from alpha_agent.schemas import AgentResponse, ToolCall, ToolResult

    results = []
    for tc in (tool_calls or []):
        results.append(
            ToolResult(
                tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments", {})),
                output=tc.get("output", "{}"),
                success=tc.get("success", True),
                error=tc.get("error"),
            )
        )
    return AgentResponse(
        query="",
        final_answer=final_answer,
        tool_calls=results,
        data={"blocked": blocked} if blocked else {},
    )


@pytest.fixture(scope="module")
def golden_set() -> list[EvalCase]:
    """The real golden set shipped with the repo."""
    path = Path(__file__).resolve().parent.parent / "data" / "golden" / "golden_set.json"
    return load_golden_set(path)


# ---------------------------------------------------------------------------
# Golden set loader
# ---------------------------------------------------------------------------


class TestGoldenSetLoader:
    def test_loads_cases_from_disk(self, golden_set: list[EvalCase]) -> None:
        assert len(golden_set) >= 10
        assert all(isinstance(c, EvalCase) for c in golden_set)

    def test_requires_query(self) -> None:
        with pytest.raises(ValueError):
            EvalCase.model_validate({"query": "  ", "category": "price"})

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_golden_set("does/not/exist.json")

    def test_load_non_list_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"query": "x"}', encoding="utf-8")
        with pytest.raises(ValueError):
            load_golden_set(bad)

    def test_includes_all_expected_categories(self, golden_set: list[EvalCase]) -> None:
        categories = {c.category for c in golden_set}
        assert {"price", "multiple-tools", "failure-mode", "edge-case", "malicious"} <= categories

    def test_has_malicious_cases(self, golden_set: list[EvalCase]) -> None:
        malicious = [c for c in golden_set if c.is_malicious]
        assert len(malicious) >= 1
        assert all(c.should_block for c in malicious)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_captures_tool_calls(self, golden_set: list[EvalCase]) -> None:
        price_case = next(c for c in golden_set if c.category == "price")
        fake = FakeAgent(
            _agent_response(
                final_answer="AAPL is trading at 185.50.",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "arguments": {"symbol": "AAPL"},
                        "output": '{"symbol": "AAPL", "price": 185.50}',
                    }
                ],
            )
        )
        judge = MockJudge()
        results = run_evals(fake, [price_case], judge)

        assert len(results) == 1
        r = results[0]
        assert "get_stock_price" in r.called_tools
        assert r.used_expected_tool is True
        assert r.used_expected_symbol is True

    def test_detects_wrong_tool(self, golden_set: list[EvalCase]) -> None:
        price_case = next(c for c in golden_set if c.category == "price")
        fake = FakeAgent(
            _agent_response(
                final_answer="Some answer",
                tool_calls=[{"name": "get_company_info", "arguments": {"symbol": "AAPL"}, "output": "{}"}],
            )
        )
        results = run_evals(fake, [price_case], MockJudge())
        assert results[0].used_expected_tool is False
        assert results[0].passed is False

    def test_malicious_blocked_passes(self, golden_set: list[EvalCase]) -> None:
        malicious = next(c for c in golden_set if c.is_malicious)
        fake = FakeAgent(_agent_response(final_answer="Blocked by guardrails.", blocked=True))
        results = run_evals(fake, [malicious], MockJudge())
        assert results[0].was_blocked is True
        assert results[0].passed is True

    def test_malicious_not_blocked_fails(self, golden_set: list[EvalCase]) -> None:
        malicious = next(c for c in golden_set if c.is_malicious)
        fake = FakeAgent(_agent_response(final_answer="I'll help you anyway.", blocked=False))
        results = run_evals(fake, [malicious], MockJudge())
        assert results[0].was_blocked is False
        assert results[0].passed is False

    def test_evidence_capture_grounding(self, golden_set: list[EvalCase]) -> None:
        price_case = next(c for c in golden_set if c.category == "price")
        fake = FakeAgent(
            _agent_response(
                final_answer="The stock price is 185.50.",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "arguments": {"symbol": "AAPL"},
                        "output": '{"symbol": "AAPL", "price": 185.50}',
                    }
                ],
            )
        )
        results = run_evals(fake, [price_case], MockJudge())
        r = results[0]
        # price & symbol appear in output/answer -> grounded.
        assert r.criteria.grounded == 1.0


class TestEdgeCaseMissingSymbol:
    """CASO 7: a query with no ticker must produce a clarification request.

    The correct behaviour is to ask the user for the ticker WITHOUT calling
    any tool or inventing a symbol.  The eval must therefore FAIL an agent
    that calls a tool instead of asking.
    """

    def _edge_case(self, golden_set: list[EvalCase]) -> EvalCase:
        return next(
            c for c in golden_set
            if c.category == "edge-case" and "stock price" in c.query
        )

    def test_case_is_annotated_as_clarification(self, golden_set: list[EvalCase]) -> None:
        case = self._edge_case(golden_set)
        # No tool is expected: calling any tool is a failure.
        assert case.expected_tool is None
        assert case.expected_symbol is None
        # Grounding comes from the clarification wording, not price data.
        assert "ticker" in case.min_evidence
        assert "symbol" in case.min_evidence
        assert "error" not in case.min_evidence
        assert "price" not in case.min_evidence

    def test_passes_when_agent_asks_for_ticker_without_tools(self, golden_set: list[EvalCase]) -> None:
        case = self._edge_case(golden_set)
        fake = FakeAgent(
            _agent_response(
                final_answer=(
                    "I need the name of the company or its ticker symbol to provide "
                    "you with the stock price. Could you please specify which stock "
                    "you are interested in?"
                ),
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        assert r.called_tools == []
        assert r.used_expected_tool is True
        assert r.passed is True

    def test_fails_when_agent_calls_a_tool_instead(self, golden_set: list[EvalCase]) -> None:
        case = self._edge_case(golden_set)
        fake = FakeAgent(
            _agent_response(
                final_answer="AAPL is trading at 185.50.",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "arguments": {"symbol": "AAPL"},
                        "output": '{"symbol": "AAPL", "price": 185.50}',
                    }
                ],
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        assert r.used_expected_tool is False
        assert any("no tool should be called" in d for d in r.details)
        assert r.passed is False

    def test_fails_when_answer_does_not_ask_for_ticker(self, golden_set: list[EvalCase]) -> None:
        case = self._edge_case(golden_set)
        fake = FakeAgent(_agent_response(final_answer="I cannot help with that."))
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        assert r.called_tools == []
        assert any("missing evidenced fields" in d for d in r.details)
        assert r.passed is False


# ---------------------------------------------------------------------------
# Mock judge
# ---------------------------------------------------------------------------


class TestMockJudge:
    def test_grounded_all_evidence_present(self) -> None:
        ctx = JudgeContext(
            query="price of AAPL",
            final_answer="AAPL is 185.50",
            min_evidence=["price", "symbol"],
            tool_data=[{"symbol": "AAPL", "price": 185.50}],
        )
        criteria = MockJudge().evaluate(ctx)
        assert criteria.grounded == 1.0

    def test_grounded_missing_evidence_scores_zero(self) -> None:
        # Evidence keys ("earnings", "volume") are absent from both the
        # answer and the tool data -> grounded must be 0.0.
        ctx = JudgeContext(
            query="price of AAPL",
            final_answer="AAPL is 200.00",
            min_evidence=["earnings", "volume"],
            tool_data=[{"symbol": "AAPL", "price": 185.50}],
        )
        criteria = MockJudge().evaluate(ctx)
        assert criteria.grounded == 0.0

    def test_correct_tool_matches(self) -> None:
        ctx = JudgeContext(query="q", final_answer="a", called_tools=["get_stock_price"], expected_tool="get_stock_price")
        assert MockJudge().evaluate(ctx).correct_tool == 1.0

    def test_correct_tool_mismatch(self) -> None:
        ctx = JudgeContext(query="q", final_answer="a", called_tools=["search_news"], expected_tool="get_stock_price")
        assert MockJudge().evaluate(ctx).correct_tool == 0.0

    def test_multiple_tool_expectation(self) -> None:
        ctx = JudgeContext(query="q", final_answer="a", called_tools=["get_stock_price", "get_stock_history"], expected_tool="multiple")
        assert MockJudge().evaluate(ctx).correct_tool == 1.0

    def test_weighted_score_in_bounds(self) -> None:
        criteria = JudgeCriteria(grounded=1.0, correct_tool=1.0, clarity=0.5, relevance=1.0)
        score = criteria.weighted_score()
        assert 0.0 <= score <= 1.0

    def test_weighted_score_value(self) -> None:
        criteria = JudgeCriteria(grounded=1.0, correct_tool=1.0, clarity=1.0, relevance=1.0)
        assert criteria.weighted_score() == 1.0


# ---------------------------------------------------------------------------
# LLMJudge._parse_score — robust JSON parsing
# ---------------------------------------------------------------------------


class TestLLMJudgeParseScore:
    """Verify _parse_score handles various LLM output formats."""

    def test_parse_direct_json(self) -> None:
        raw = '{"grounded": 0.9, "correct_tool": 1.0, "clarity": 0.8, "relevance": 0.7}'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.9
        assert c.correct_tool == 1.0
        assert c.clarity == 0.8
        assert c.relevance == 0.7

    def test_parse_json_with_surrounding_text(self) -> None:
        raw = 'Here is my evaluation:\n{"grounded": 0.85, "correct_tool": 0.9, "clarity": 0.75, "relevance": 0.95}\nHope this helps.'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.85
        assert c.correct_tool == 0.9
        assert c.clarity == 0.75
        assert c.relevance == 0.95

    def test_parse_json_in_markdown_fences(self) -> None:
        raw = '```json\n{"grounded": 0.6, "correct_tool": 0.7, "clarity": 0.8, "relevance": 0.9}\n```'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.6
        assert c.correct_tool == 0.7
        assert c.clarity == 0.8
        assert c.relevance == 0.9

    def test_parse_json_in_fences_with_text(self) -> None:
        raw = 'Based on my analysis, here are the scores:\n```json\n{"grounded": 0.5, "correct_tool": 0.6, "clarity": 0.7, "relevance": 0.8}\n```\nThese scores reflect the quality.'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.5
        assert c.correct_tool == 0.6

    def test_parse_fallback_to_zeros(self) -> None:
        """Completely non-JSON input returns all zeros."""
        c = LLMJudge._parse_score("I cannot provide scores at this time.")
        assert c.grounded == 0.0
        assert c.correct_tool == 0.0
        assert c.clarity == 0.0
        assert c.relevance == 0.0

    def test_parse_empty_string(self) -> None:
        c = LLMJudge._parse_score("")
        assert c.grounded == 0.0

    def test_parse_json_without_fence_wrapper(self) -> None:
        """Markdown fences without the json language tag."""
        raw = '```\n{"grounded": 0.3, "correct_tool": 0.4, "clarity": 0.5, "relevance": 0.6}\n```'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.3
        assert c.correct_tool == 0.4

    def test_parse_clamps_out_of_range(self) -> None:
        raw = '{"grounded": 1.5, "correct_tool": -0.1, "clarity": 0.5, "relevance": 0.5}'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 1.0  # clamped
        assert c.correct_tool == 0.0  # clamped

    def test_parse_double_brace_wrapped(self) -> None:
        """LLM returns {{…}} (double curly) around the JSON payload."""
        raw = '{{"grounded": 0.5, "correct_tool": 1.0, "clarity": 0.8, "relevance": 0.0}}'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.5
        assert c.correct_tool == 1.0
        assert c.clarity == 0.8
        assert c.relevance == 0.0

    def test_parse_double_brace_with_surrounding_text(self) -> None:
        """Double-brace JSON preceded/followed by free text."""
        raw = 'prefix {{"grounded": 0.7, "correct_tool": 1.0, "clarity": 0.9, "relevance": 0.8}} suffix'
        c = LLMJudge._parse_score(raw)
        assert c.grounded == 0.7
        assert c.correct_tool == 1.0
        assert c.clarity == 0.9
        assert c.relevance == 0.8


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_aggregates_counts(self, golden_set: list[EvalCase]) -> None:
        # Use the AAPL price case, which the fake agent answers correctly so
        # the report aggregates a genuine all-pass run.
        case = next(c for c in golden_set if c.category == "price" and c.expected_symbol == "AAPL")
        fake = FakeAgent(
            _agent_response(
                final_answer="AAPL is trading at 185.50 USD.",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "arguments": {"symbol": "AAPL"},
                        "output": '{"symbol": "AAPL", "price": 185.50, "change_pct": 1.23}',
                    }
                ],
            )
        )
        results = run_evals(fake, [case], MockJudge())
        report = EvalReport.from_results(results)

        assert report.total == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.pass_rate == 1.0
        assert report.criteria_breakdown["grounded"] == 1.0
        assert results[0].passed is True

    def test_report_breakdown_per_category(self, golden_set: list[EvalCase]) -> None:
        cases = [c for c in golden_set if c.category in {"price", "failure-mode"}]
        results = run_evals(FakeAgent(_agent_response(final_answer="answer")), cases, MockJudge())
        report = EvalReport.from_results(results)
        assert "price" in report.category_breakdown
        assert "failure-mode" in report.category_breakdown

    def test_save_and_reload_report(self, golden_set: list[EvalCase], tmp_path: Path) -> None:
        cases = [c for c in golden_set if c.category == "price"]
        results = run_evals(FakeAgent(_agent_response(final_answer="a")), cases, MockJudge())
        report = EvalReport.from_results(results)

        out = tmp_path / "report_test.json"
        save_report(report, out)
        assert out.exists()

        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total"] == len(cases)
        assert "criteria_breakdown" in data
        assert len(data["results"]) == len(cases)


# ---------------------------------------------------------------------------
# Judge calibration — error-handling and clarification cases
# ---------------------------------------------------------------------------


class TestFailureModeCalibration:
    """Verify the judge correctly scores failure-mode (error-handling) cases.

    When the agent calls a tool with an invalid symbol and honestly reports
    the error (without fabricating data), the judge must NOT penalise
    grounded, correct_tool, or relevance.
    """

    def test_mock_judge_failure_mode_honest_error(self, golden_set: list[EvalCase]) -> None:
        """CASO 4: agent calls get_stock_history, gets failure, honestly reports it."""
        case = next(
            c for c in golden_set
            if c.category == "failure-mode" and "INVALIDTICK123" in c.query
        )
        fake = FakeAgent(
            _agent_response(
                final_answer=(
                    'It seems that "INVALIDTICK123" is not a valid ticker symbol. '
                    "Could you please provide the correct ticker symbol for the stock "
                    "you're interested in?"
                ),
                tool_calls=[
                    {
                        "name": "get_stock_history",
                        "arguments": {"symbol": "INVALIDTICK123"},
                        "output": '{"error": "No data found for symbol INVALIDTICK123"}',
                    }
                ],
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        # Honest error response → grounded must be high (no fabricated data)
        assert r.criteria.grounded == 1.0, (
            f"grounded={r.criteria.grounded}: honest error should score high"
        )
        # Expected tool was called → correct_tool must be 1.0
        assert r.criteria.correct_tool == 1.0, (
            f"correct_tool={r.criteria.correct_tool}: get_stock_history was called"
        )
        # Response addresses the query → relevance must be high
        assert r.criteria.relevance == 1.0, (
            f"relevance={r.criteria.relevance}: error response addresses the query"
        )
        assert r.criteria.clarity == 1.0
        assert r.passed is True

    def test_mock_judge_failure_mode_fabricated_data(self, golden_set: list[EvalCase]) -> None:
        """Agent incorrectly fabricates a price for an invalid ticker → must fail."""
        case = next(
            c for c in golden_set
            if c.category == "failure-mode" and "INVALIDTICK123" in c.query
        )
        fake = FakeAgent(
            _agent_response(
                final_answer="INVALIDTICK123 is trading at 42.50 USD.",
                tool_calls=[
                    {
                        "name": "get_stock_history",
                        "arguments": {"symbol": "INVALIDTICK123"},
                        "output": '{"error": "No data found for symbol INVALIDTICK123"}',
                    }
                ],
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        # Fabricated price with no evidence → grounded must be low
        assert r.criteria.grounded == 0.0, (
            f"grounded={r.criteria.grounded}: fabricated price should score 0"
        )
        assert r.passed is False

    def test_failure_mode_case_category_propagated(self, golden_set: list[EvalCase]) -> None:
        """JudgeContext for failure-mode cases carries the category."""
        case = next(c for c in golden_set if c.category == "failure-mode")
        fake = FakeAgent(_agent_response(final_answer="test"))
        results = run_evals(fake, [case], MockJudge())
        # Verify the EvalResult has the right case with category
        assert results[0].case.category == "failure-mode"


class TestEdgeCaseCalibration:
    """Verify the judge correctly scores edge-case (clarification) cases.

    When the agent correctly asks for missing information instead of calling
    tools, the judge must NOT penalise correct_tool or grounded.
    """

    def _edge_case(self, golden_set: list[EvalCase]) -> EvalCase:
        return next(
            c for c in golden_set
            if c.category == "edge-case" and "stock price" in c.query
        )

    def test_mock_judge_clarification_no_tool_no_fabrication(self, golden_set: list[EvalCase]) -> None:
        """CASO 7: agent asks for the ticker without calling any tool."""
        case = self._edge_case(golden_set)
        fake = FakeAgent(
            _agent_response(
                final_answer=(
                    "I need the ticker symbol to find the stock price. Could you "
                    "please provide the name or ticker symbol of the company you're "
                    "interested in?"
                ),
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        # No tool calls was the correct behaviour → correct_tool must be 1.0
        assert r.criteria.correct_tool == 1.0, (
            f"correct_tool={r.criteria.correct_tool}: not calling tools was correct"
        )
        # Honest clarification with no fabricated data → grounded must be high
        assert r.criteria.grounded == 1.0, (
            f"grounded={r.criteria.grounded}: clarification with no fabrication"
        )
        # Response asks for the missing info → relevance must be high
        assert r.criteria.relevance == 1.0, (
            f"relevance={r.criteria.relevance}: asks for the missing ticker"
        )
        assert r.criteria.clarity == 1.0
        assert r.passed is True

    def test_mock_judge_clarification_tool_called_fails(self, golden_set: list[EvalCase]) -> None:
        """Agent calls a tool when it should ask for the ticker → must fail."""
        case = self._edge_case(golden_set)
        fake = FakeAgent(
            _agent_response(
                final_answer="AAPL is trading at 185.50.",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "arguments": {"symbol": "AAPL"},
                        "output": '{"symbol": "AAPL", "price": 185.50}',
                    }
                ],
            )
        )
        results = run_evals(fake, [case], MockJudge())
        r = results[0]
        # Called a tool when none was expected → correct_tool must be 0.0
        assert r.criteria.correct_tool == 0.0, (
            f"correct_tool={r.criteria.correct_tool}: tool was called when none expected"
        )
        assert r.used_expected_tool is False
        assert r.passed is False

    def test_edge_case_case_category_propagated(self, golden_set: list[EvalCase]) -> None:
        """JudgeContext for edge-case cases carries the category."""
        case = self._edge_case(golden_set)
        fake = FakeAgent(_agent_response(final_answer="test"))
        results = run_evals(fake, [case], MockJudge())
        assert results[0].case.category == "edge-case"


class TestLLMJudgePromptCalibration:
    """Verify the LLMJudge system prompt contains category-aware rules."""

    def test_prompt_includes_failure_mode_rules(self) -> None:
        prompt = LLMJudge._JUDGE_PROMPT
        assert "failure-mode" in prompt.lower() or "failure mode" in prompt.lower()
        assert "error" in prompt.lower() or "honestly" in prompt.lower()

    def test_prompt_includes_edge_case_rules(self) -> None:
        prompt = LLMJudge._JUDGE_PROMPT
        assert "edge-case" in prompt.lower() or "edge case" in prompt.lower()
        assert "clarification" in prompt.lower()

    def test_prompt_instructs_grounded_for_error_responses(self) -> None:
        """The prompt must tell the judge that honest error responses are grounded."""
        prompt = LLMJudge._JUDGE_PROMPT
        # Should mention that fabricated data is bad but honest errors are good
        assert "invent" in prompt.lower() or "fabricat" in prompt.lower()

    def test_prompt_instructs_no_tool_for_clarification(self) -> None:
        """The prompt must tell the judge that not calling tools is correct for edge cases."""
        prompt = LLMJudge._JUDGE_PROMPT
        assert "None" in prompt or "no tool" in prompt.lower()


class TestCategoryInJudgeContext:
    """Verify that JudgeContext correctly carries the category field."""

    def test_category_defaults_to_none(self) -> None:
        ctx = JudgeContext(query="q", final_answer="a")
        assert ctx.category is None

    def test_category_can_be_set(self) -> None:
        ctx = JudgeContext(query="q", final_answer="a", category="failure-mode")
        assert ctx.category == "failure-mode"

    def test_category_none_does_not_break_mock_judge(self) -> None:
        """MockJudge works fine without a category (backward compat)."""
        ctx = JudgeContext(
            query="q",
            final_answer="This is a test answer with several words in it.",
            min_evidence=["test"],
        )
        criteria = MockJudge().evaluate(ctx)
        assert criteria.grounded == 1.0  # "test" appears in the answer
        assert criteria.correct_tool == 1.0  # no expected_tool, no called_tools
        assert criteria.clarity == 1.0
        assert criteria.relevance == 1.0
