"""LLM-as-a-judge.

Scores an agent's answer against four quality criteria using a local LLM
(Ollama), and aggregates them into a weighted global score.

The judge also ships a deterministic ``MockJudge`` so tests and the
``--mock`` CLI flag can run without a live model.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient

logger = logging.getLogger(__name__)

# Per-criterion weights used to compute the weighted global score.
# These sum to 1.0 and reflect the relative importance of each criterion.
DEFAULT_WEIGHTS: dict[str, float] = {
    "grounded": 0.35,
    "correct_tool": 0.30,
    "clarity": 0.15,
    "relevance": 0.20,
}


class JudgeContext(BaseModel):
    """Facts the judge needs to score an answer.

    Attributes
    ----------
    query:
        The original user query.
    final_answer:
        The agent's final answer text.
    called_tools:
        Names of the tools actually invoked by the agent (in order).
    category:
        The golden-case category (e.g. ``"price"``, ``"failure-mode"``,
        ``"edge-case"``, ``"malicious"``).  Used by the LLM judge to
        apply category-aware scoring rules.
    expected_tool:
        The tool annotated in the golden case as expected.
    expected_symbol:
        The symbol annotated in the golden case as expected.
    min_evidence:
        Keys/terms the answer should contain to be considered grounded.
    tool_data:
        The raw dicts returned by the agent's tools (for grounding checks).
    """

    query: str
    final_answer: str
    called_tools: list[str] = Field(default_factory=list)
    category: str | None = None
    expected_tool: str | None = None
    expected_symbol: str | None = None
    min_evidence: list[str] = Field(default_factory=list)
    tool_data: list[dict] = Field(default_factory=list)


class JudgeCriteria(BaseModel):
    """Scores (0-1) for each evaluation criterion."""

    grounded: float = Field(ge=0.0, le=1.0)
    correct_tool: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)

    def weighted_score(self, weights: dict[str, float] | None = None) -> float:
        """Return the weighted global score in [0, 1].

        Parameters
        ----------
        weights:
            Optional dict of criterion -> weight (must sum to 1.0).  Defaults
            to :data:`DEFAULT_WEIGHTS`.
        """
        w = weights if weights is not None else DEFAULT_WEIGHTS
        total = (
            self.grounded * w.get("grounded", 0.0)
            + self.correct_tool * w.get("correct_tool", 0.0)
            + self.clarity * w.get("clarity", 0.0)
            + self.relevance * w.get("relevance", 0.0)
        )
        return round(float(total), 4)

    def to_dict(self) -> dict[str, float]:
        """Return the criteria as a plain float dict."""
        return {
            "grounded": self.grounded,
            "correct_tool": self.correct_tool,
            "clarity": self.clarity,
            "relevance": self.relevance,
        }


class BaseJudge(ABC):
    """Abstract judge that scores an answer against :data:`JudgeContext`."""

    @abstractmethod
    def evaluate(self, ctx: JudgeContext) -> JudgeCriteria:
        """Return a :class:`JudgeCriteria` for *ctx*."""


class LLMJudge(BaseJudge):
    """Scores answers using a local LLM (Ollama).

    Parameters
    ----------
    model:
        Ollama model tag to use as the judge.
    base_url:
        Reachable Ollama address (Docker network, never ``localhost``).
    weights:
        Optional criteria weights; defaults to :data:`DEFAULT_WEIGHTS`.
    timeout:
        Per-request timeout in seconds.
    """

    _JUDGE_PROMPT = """\
You are an impartial evaluation judge for an AI assistant with financial
tools.  Score the assistant's answer on four criteria using ONLY the
provided facts.  A CASE CATEGORY is included so you can adapt scoring to
the type of scenario.

RESPOND WITH A SINGLE JSON OBJECT with float keys between 0.0 and 1.0:
{{"grounded": <float>, "correct_tool": <float>, "clarity": <float>, "relevance": <float>}}

SCORING RULES (general):
- "grounded": 1.0 if every number/fact in the answer can be traced to the
  tool output (or the answer contains no numbers at all).  Penalise
  invented numbers or figures not present in the data.
- "correct_tool": 1.0 if the assistant's tool-calling behaviour matches
  the expectation for this case.  When EXPECTED TOOL is "None", the
  assistant must NOT call any tool (score 1.0 for zero tool calls;
  score 0.0 if tools were called).
- "clarity": 1.0 if the answer is clear, concise and well-structured.
- "relevance": 1.0 if the answer addresses the user's intent.

RULES FOR ERROR-HANDLING / FAILURE-MODE CASES (category = "failure-mode"):
These cases test whether the agent handles invalid/missing data gracefully.
- "grounded": HIGH (0.8-1.0) if the response honestly communicates the
  error (e.g. "not a valid ticker", "could not retrieve", "no data
  available") WITHOUT inventing any numbers or figures.  A correct
  error-handling response that honestly asks the user to fix their input
  is well-grounded.  LOW (0.0-0.2) only if the response fabricates
  data as if the call succeeded.
- "correct_tool": 1.0 if the expected tool WAS invoked (the agent tried
  to look up data and failed gracefully), or if no tool was expected
  and none was called.
- "relevance": HIGH (0.8-1.0) if the response directly addresses the
  user's query by explaining the failure and what to do next (e.g.
  asking for the correct ticker).  LOW (0.0-0.3) only if the response
  ignores the query entirely.

RULES FOR EDGE-CASE / CLARIFICATION CASES (category = "edge-case"):
These cases test whether the agent asks for missing information instead
of guessing or calling tools with incomplete input.
- "grounded": HIGH (0.8-1.0) if the response honestly asks for the
  missing information (e.g. "please provide the ticker symbol") without
  inventing any data.  A correct clarification request contains no
  fabricated numbers and is therefore well-grounded.
- "correct_tool": 1.0 if NO tools were called (the correct behaviour
  is to ask the user).  Score 0.0 only if tools were called when the
  case expected the agent to ask for clarification.
- "relevance": 1.0 if the response asks specifically for the information
  it needs to answer the original query (e.g. the missing ticker symbol).

Do not output any text outside the JSON object.
"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        weights: dict[str, float] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self._client = OllamaClient(base_url=base_url, model=model, timeout=timeout)

    def evaluate(self, ctx: JudgeContext) -> JudgeCriteria:
        """Score *ctx* by asking the LLM to rate the criteria."""
        tool_outputs = json.dumps(ctx.tool_data, default=str)
        messages = [
            {"role": "system", "content": self._JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"USER QUERY: {ctx.query}\n\n"
                    f"CASE CATEGORY: {ctx.category or 'N/A'}\n"
                    f"EXPECTED TOOL: {ctx.expected_tool or 'N/A'}\n"
                    f"EXPECTED SYMBOL: {ctx.expected_symbol or 'N/A'}\n"
                    f"TOOLS CALLED: {json.dumps(ctx.called_tools)}\n"
                    f"TOOL OUTPUT: {tool_outputs}\n\n"
                    f"ASSISTANT ANSWER:\n{ctx.final_answer}\n"
                ),
            },
        ]
        resp = self._client.chat(messages, temperature=0.0)
        return self._parse_score(resp.content)

    def close(self) -> None:
        """Close the underlying LLM client."""
        self._client.close()

    @staticmethod
    def _parse_score(content: str) -> JudgeCriteria:
        """Parse the judge's JSON answer into a :class:`JudgeCriteria`.

        Uses a cascading parser to handle common LLM formatting issues:
        1. Direct ``json.loads`` of the full content.
        2. Brace-matching to extract the outermost ``{…}`` substring.
        3. Markdown fenced blocks (````json … ````).
        4. Fallback to all zeros so a single badly-formatted response never
           crashes the whole eval.
        """

        def _clamp(value) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        def _from_dict(data: dict) -> JudgeCriteria:
            return JudgeCriteria(
                grounded=_clamp(data.get("grounded", 0.0)),
                correct_tool=_clamp(data.get("correct_tool", 0.0)),
                clarity=_clamp(data.get("clarity", 0.0)),
                relevance=_clamp(data.get("relevance", 0.0)),
            )

        data: dict | None = None

        # Attempt 1: direct parse
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # Attempt 2: brace-matching — find first '{' and last '}'
        if data is None:
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                data = json.loads(content[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

        # Attempt 2b: strip one extra level of wrapped braces — the model
        # sometimes returns {{…}} (double curly) around the JSON payload.
        if data is None:
            try:
                start = content.index("{")
                end = content.rindex("}")
                inner = content[start + 1 : end]  # peel one outer {…}
                data = json.loads(inner)
            except (ValueError, json.JSONDecodeError, IndexError):
                pass

        # Attempt 3: extract from ```json ... ``` fences
        if data is None:
            import re

            fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
            if fence:
                try:
                    data = json.loads(fence.group(1).strip())
                except (json.JSONDecodeError, TypeError):
                    pass

        if data is None or not isinstance(data, dict):
            logger.warning("Judge LLM returned non-JSON; scoring all zeros")
            return JudgeCriteria(grounded=0.0, correct_tool=0.0, clarity=0.0, relevance=0.0)

        return _from_dict(data)


class MockJudge(BaseJudge):
    """Deterministic judge for tests and offline runs.

    Computes criteria from the golden-case annotations rather than an LLM, so
    results are stable and cheap.  Intended for CI smoke tests and for
    validating the eval harness itself, not as a substitute for a real LLM
    judge in production.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS

    def evaluate(self, ctx: JudgeContext) -> JudgeCriteria:
        """Compute a deterministic score from *ctx*."""
        grounded = self._score_grounded(ctx)
        correct_tool = self._score_correct_tool(ctx)
        clarity = self._score_clarity(ctx)
        relevance = self._score_relevance(ctx)
        return JudgeCriteria(
            grounded=grounded,
            correct_tool=correct_tool,
            clarity=clarity,
            relevance=relevance,
        )

    # --- deterministic heuristics -------------------------------------

    @staticmethod
    def _score_grounded(ctx: JudgeContext) -> float:
        """1.0 if all required evidence keys appear in the answer, else 0.0."""
        if not ctx.min_evidence:
            return 1.0
        combined = " ".join(
            [ctx.final_answer] + [str(d) for d in ctx.tool_data]
        ).lower()
        matched = [ev for ev in ctx.min_evidence if str(ev).lower() in combined]
        return 1.0 if len(matched) >= len(ctx.min_evidence) else 0.0

    @staticmethod
    def _score_correct_tool(ctx: JudgeContext) -> float:
        """1.0 if the agent invoked the expected tool (or all needed ones).

        ``expected_tool=None`` means no tool should be called: the score is
        1.0 only when the agent made zero tool calls (e.g. a clarification
        request or a blocked query).
        """
        if ctx.expected_tool == "multiple":
            return 1.0 if len(ctx.called_tools) >= 2 else 0.0
        if not ctx.expected_tool:
            return 1.0 if not ctx.called_tools else 0.0
        return 1.0 if ctx.expected_tool in ctx.called_tools else 0.0

    @staticmethod
    def _score_clarity(ctx: JudgeContext) -> float:
        """Heuristic: prefer answers of reasonable length with structure."""
        answer = ctx.final_answer.strip()
        if not answer:
            return 0.0
        words = len(answer.split())
        if 5 <= words <= 120:
            return 1.0
        return 0.5

    @staticmethod
    def _score_relevance(ctx: JudgeContext) -> float:
        """Heuristic: answer should not be empty and should contain evidence."""
        if not ctx.final_answer.strip():
            return 0.0
        if ctx.min_evidence:
            combined = " ".join(
                [ctx.final_answer] + [str(d) for d in ctx.tool_data]
            ).lower()
            if any(str(ev).lower() in combined for ev in ctx.min_evidence):
                return 1.0
            return 0.5
        return 1.0
