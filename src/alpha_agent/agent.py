"""AlphaAgent – a ReAct-style agent with real function-calling.

The agent follows the thought → tool-decision → execution → observation
loop until the LLM produces a final answer or the iteration budget is
exhausted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .llm import OllamaClient
from .schemas import AgentResponse, Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are AlphaAgent, a helpful AI assistant with access to financial tools.

## Tool selection rules (follow strictly)

1. **get_stock_price** — Use when the user asks for the CURRENT price,
   current value, or how a stock is trading RIGHT NOW, and no time range
   is mentioned.  Examples: "What is AAPL's price?", "How much is MSFT
   trading at?"

2. **get_stock_history** — Use when the user asks about a stock's price
   or performance OVER A PERIOD OF TIME.  Any mention of a time range
   (e.g. "last month", "over the past week", "year-to-date", "last 3
   months", "historical", "past year", "1-month history", "weekly",
   "daily", "since January") means you MUST call get_stock_history, NOT
   get_stock_price and NOT search_news.  Examples: "What was TSLA over
   the last month?", "Show me NVDA history for the past year."

3. **search_news** — Use ONLY when the user explicitly asks for news,
   articles, headlines, or recent events about a company.  Do NOT use
   this tool as a fallback when a price-history query fails or when
   there is no ticker symbol.

4. **calculate_risk_metrics** — Use when the user asks about risk,
   volatility, Sharpe ratio, or drawdown.

5. **get_company_info** — Use when the user asks about company
   fundamentals (sector, industry, market cap, P/E, description).

## Missing symbol handling

- If the user's query does NOT contain an identifiable ticker symbol
  (e.g. "What is the stock price?" with no company name or ticker), do
  NOT call any tool.  Instead, ask the user to specify the ticker
  symbol.  A ticker is an uppercase code like AAPL, MSFT, GOOG, TSLA,
  or a variant like ^GSPC, BTC-USD, BRK-B.

- If the query contains a ticker-like code (even one you suspect is
  wrong), ALWAYS call the appropriate tool for the request:
  get_stock_history when a time range is mentioned, get_stock_price
  for the current price, etc.  Never judge a ticker's validity
  yourself: string format is not proof of validity.  The tool is the
  source of truth and returns a structured {"error": ...} for unknown
  symbols, so call it and let the tool validate.

- When a tool returns {"error": ...}, relay that exact real error to
  the user in your final answer.  Do not claim a ticker is invalid
  based on your own reasoning - only repeat what the tool reported.

## General behaviour

- Call the appropriate tool for the request.  You will receive the
  tool's output and can then decide what to do next.
- When you have enough information to answer, respond directly without
  calling more tools.
- Always show your reasoning step by step before giving a final answer."""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AlphaAgent:
    """ReAct agent backed by Ollama function calling.

    Parameters
    ----------
    llm_client:
        An :class:`~alpha_agent.llm.OllamaClient` instance.
    tools:
        List of :class:`~alpha_agent.schemas.Tool` objects the agent may use.
    max_iterations:
        Hard cap on the thought→action→observation loop to prevent runaway
        execution.  Each iteration allows **one** tool call.
    system_prompt:
        Override the default system prompt.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        tools: list[Tool] | None = None,
        max_iterations: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm_client
        self.tools: list[Tool] = tools or []
        self.tool_map: dict[str, Tool] = {t.name: t for t in self.tools}
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, query: str) -> AgentResponse:
        """Run the ReAct loop for *query* and return an :class:`AgentResponse`.

        The loop terminates when:
        1. The LLM produces a text answer **without** tool calls (final answer).
        2. The iteration budget is exhausted.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        reasoning: list[str] = []
        tool_results: list[ToolResult] = []
        final_answer = ""
        data: dict[str, Any] = {}

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Iteration %d/%d", iteration, self.max_iterations)

            # --- Ask the LLM ---
            llm_resp = self.llm.chat_with_tools(messages, self.tools)

            # Record any reasoning the model emitted
            if llm_resp.content:
                reasoning.append(llm_resp.content)

            # --- No tool calls → final answer ---
            if not llm_resp.tool_calls:
                final_answer = llm_resp.content
                break

            # --- Execute each requested tool ---
            for tc in llm_resp.tool_calls:
                result = self._execute_tool(tc)
                tool_results.append(result)

                # Append the tool result to the conversation so the LLM
                # can observe the outcome in the next iteration.
                messages.append({
                    "role": "assistant",
                    "content": llm_resp.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "content": result.output if result.success else f"ERROR: {result.error}",
                })

                # If a tool returned structured data, collect it
                try:
                    parsed = json.loads(result.output)
                    if isinstance(parsed, dict):
                        data.update(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            # Loop exhausted without a final answer
            final_answer = (
                "I was unable to complete the analysis within the allowed "
                "number of iterations. Here is what I found so far:\n\n"
                + "\n".join(reasoning)
            )

        return AgentResponse(
            query=query,
            final_answer=final_answer,
            reasoning=reasoning,
            tool_calls=tool_results,
            iterations_used=min(iteration, self.max_iterations),
            data=data,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tc: ToolCall) -> ToolResult:
        """Run a single tool call and capture the outcome."""
        tool = self.tool_map.get(tc.name)
        if tool is None:
            return ToolResult(
                tool_call=tc,
                output="",
                success=False,
                error=f"Unknown tool: {tc.name}",
            )

        if tool.fn is None:
            return ToolResult(
                tool_call=tc,
                output="",
                success=False,
                error=f"Tool '{tc.name}' has no callable function attached.",
            )

        try:
            output = tool.fn(**tc.arguments)
            # Serialize structured outputs (dict/list) as JSON so consuming
            # code (e.g. grounding checks) can parse them reliably.
            if isinstance(output, (dict, list)):
                serialized = json.dumps(output, default=str)
            else:
                serialized = str(output)
            return ToolResult(
                tool_call=tc,
                output=serialized,
                success=True,
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tc.name)
            return ToolResult(
                tool_call=tc,
                output="",
                success=False,
                error=str(exc),
            )
