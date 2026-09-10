"""Guarded agent wrapper.

Wraps :class:`~alpha_agent.agent.AlphaAgent` with a configurable pipeline
of guardrails that are applied **before** (input), **after** (output), and
**around** tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .agent import AlphaAgent
from .guardrails.anti_hallucination import AntiHallucinationGuardrail
from .guardrails.base import Guardrail, ValidationResult
from .guardrails.financial import FinancialGuardrail
from .guardrails.tool_guard import ToolGuardrail
from .schemas import AgentResponse

logger = logging.getLogger(__name__)


class GuardedAlphaAgent:
    """Decorator that adds guardrails to an :class:`AlphaAgent`.

    Parameters
    ----------
    agent:
        The underlying agent instance.
    guardrails:
        Ordered list of guardrails to apply.  If ``None`` the default
        set (Financial, Tool, AntiHallucination) is used.

    Attributes
    ----------
    last_input_result : ValidationResult | None
        Result of the most recent input validation.
    last_output_result : ValidationResult | None
        Result of the most recent output validation.
    last_grounding_result : ValidationResult | None
        Result of the most recent grounding check.
    """

    def __init__(
        self,
        agent: AlphaAgent,
        guardrails: list[Guardrail] | None = None,
    ) -> None:
        self.agent = agent
        self.guardrails = guardrails if guardrails is not None else [
            FinancialGuardrail(),
            ToolGuardrail(),
            AntiHallucinationGuardrail(),
        ]

        # Typed references for convenience (populated from the list)
        self.financial: FinancialGuardrail | None = None
        self.tool_guard: ToolGuardrail | None = None
        self.anti_hallucination: AntiHallucinationGuardrail | None = None
        for g in self.guardrails:
            if isinstance(g, FinancialGuardrail):
                self.financial = g
            elif isinstance(g, ToolGuardrail):
                self.tool_guard = g
            elif isinstance(g, AntiHallucinationGuardrail):
                self.anti_hallucination = g

        # Inspection attributes
        self.last_input_result: ValidationResult | None = None
        self.last_output_result: ValidationResult | None = None
        self.last_grounding_result: ValidationResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, query: str, history: list[dict[str, Any]] | None = None) -> AgentResponse:
        """Run the guarded analysis pipeline.

        Parameters
        ----------
        query:
            The user's current question.
        history:
            Optional prior conversation messages passed through to the
            underlying agent (see :meth:`AlphaAgent.analyze`).

        Steps:
        1. **Input guardrails** – every guardrail's ``validate_input`` is
           called.  If *any* rejects, the query is blocked and a synthetic
           ``AgentResponse`` is returned without calling the LLM.

        2. **Agent execution** – the underlying ``AlphaAgent.analyze`` runs.

        3. **Output guardrails** – every guardrail's ``validate_output`` is
           called.  Financial disclaimer is injected if missing.

        4. **Grounding check** – if an ``AntiHallucinationGuardrail`` is
           present, the agent's final answer is cross-checked against the
           tool results.
        """
        # ---- 1. Input validation ----
        for guardrail in self.guardrails:
            result = guardrail.validate_input(query)
            self.last_input_result = result
            if not result.allowed:
                logger.warning("Input blocked by %s: %s", type(guardrail).__name__, result.reason)
                return AgentResponse(
                    query=query,
                    final_answer=(
                        f"I'm sorry, but I cannot process this request. "
                        f"Reason: {result.reason}"
                    ),
                    reasoning=[],
                    tool_calls=[],
                    iterations_used=0,
                    data={"blocked": True, "reason": result.reason, "details": result.details},
                )

        # ---- 2. Run the agent ----
        response = self.agent.analyze(query, history=history)

        # ---- 3. Output validation / enrichment ----
        for guardrail in self.guardrails:
            result = guardrail.validate_output(response.final_answer)
            self.last_output_result = result
            if not result.allowed:
                # FinancialGuardrail returns disclaimer text in details
                disclaimer_text = result.details.get("disclaimer_text")
                if disclaimer_text:
                    response.final_answer = response.final_answer + disclaimer_text
                    logger.info("Financial disclaimer appended to response")

        # ---- 4. Grounding verification ----
        if self.anti_hallucination is not None:
            tool_data = self._extract_tool_data(response)
            grounding = self.anti_hallucination.verify_grounding(
                response.final_answer, tool_data
            )
            self.last_grounding_result = grounding
            if not grounding.allowed:
                logger.warning("Grounding check failed: %s", grounding.reason)
                # Append a warning rather than blocking the answer
                ungrounded = grounding.details.get("ungrounded", [])
                numbers_str = ", ".join(str(u["number"]) for u in ungrounded)
                response.final_answer += (
                    f"\n\n⚠️ *Note: The following numbers could not be "
                    f"directly verified from tool data: {numbers_str}. "
                    f"Please cross-check with primary sources.*"
                )

        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_data(response: AgentResponse) -> list[dict[str, Any]]:
        """Parse JSON tool outputs into dicts for grounding checks.

        Handles both valid JSON (``json.dumps``) and Python ``repr`` format
        (``str(dict)`` with single quotes) that legacy code paths may still
        produce.
        """
        import ast

        data: list[dict[str, Any]] = []
        for tr in response.tool_calls:
            if not tr.success:
                continue
            parsed: Any = None
            # 1. Try standard JSON parsing
            try:
                parsed = json.loads(tr.output)
            except (json.JSONDecodeError, TypeError):
                pass
            # 2. Fallback: parse Python repr (single-quoted dicts/lists)
            if parsed is None and tr.output:
                try:
                    parsed = ast.literal_eval(tr.output)
                except (ValueError, SyntaxError):
                    pass
            # 3. Collect dicts from parsed result
            if isinstance(parsed, dict):
                data.append(parsed)
            elif isinstance(parsed, list):
                data.extend(item for item in parsed if isinstance(item, dict))
        return data
