"""Unit tests for AlphaAgent core — no real Ollama calls.

All external LLM interactions are mocked so these tests run fast and offline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from alpha_agent.agent import AlphaAgent
from alpha_agent.llm import OllamaClient
from alpha_agent.schemas import LLMResponse, Tool, ToolCall, ToolParameter

# ======================================================================
# Helpers
# ======================================================================


def _make_llm_response(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> LLMResponse:
    """Build a fake :class:`LLMResponse` for mocking."""
    calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in (tool_calls or [])]
    return LLMResponse(content=content, tool_calls=calls, done=True, raw={})


def _mock_client(responses: list[LLMResponse]) -> MagicMock:
    """Return a mock OllamaClient that yields *responses* in order."""
    client = MagicMock(spec=OllamaClient)
    client.chat_with_tools = MagicMock(side_effect=responses)
    return client


def _sample_tool() -> Tool:
    """A simple calculator tool for tests."""
    return Tool(
        name="calculator",
        description="Evaluate a math expression",
        parameters=[
            ToolParameter(name="expression", type="string", description="Math expression", required=True),
        ],
        fn=lambda expression: str(eval(expression)),
    )


def _failing_tool() -> Tool:
    """A tool that always raises an exception."""
    return Tool(
        name="broken_tool",
        description="Always fails",
        parameters=[],
        fn=lambda: (_ for _ in ()).throw(ValueError("deliberate failure")),
    )


def _no_fn_tool() -> Tool:
    """A tool with no callable attached."""
    return Tool(
        name="no_fn",
        description="No function",
        parameters=[],
        fn=None,
    )


# ======================================================================
# Tests: direct answer (no tools needed)
# ======================================================================


class TestDirectAnswer:
    """Agent responds directly when the LLM doesn't request any tool call."""

    def test_returns_llm_content_as_final_answer(self) -> None:
        fake_response = _make_llm_response(content="The capital of France is Paris.")
        client = _mock_client([fake_response])
        agent = AlphaAgent(llm_client=client, tools=[])

        result = agent.analyze("What is the capital of France?")

        assert result.final_answer == "The capital of France is Paris."
        assert result.tool_calls == []
        assert result.iterations_used == 1

    def test_reasoning_captured(self) -> None:
        fake_response = _make_llm_response(content="I know this directly.")
        client = _mock_client([fake_response])
        agent = AlphaAgent(llm_client=client)

        result = agent.analyze("Quick question")

        assert "I know this directly." in result.reasoning


# ======================================================================
# Tests: tool calling
# ======================================================================


class TestToolCalling:
    """Agent calls tools when the LLM requests them."""

    def test_single_tool_call(self) -> None:
        """LLM asks for a calculator tool → agent executes it → LLM answers."""
        tool = _sample_tool()
        # Round 1: LLM requests calculator
        round1 = _make_llm_response(
            content="Let me calculate that.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "2 + 3"}}],
        )
        # Round 2: LLM gives final answer after seeing tool output
        round2 = _make_llm_response(content="The answer is 5.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[tool])
        result = agent.analyze("What is 2 + 3?")

        assert result.final_answer == "The answer is 5."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].success is True
        assert result.tool_calls[0].output == "5"
        assert result.iterations_used == 2

    def test_multiple_tool_calls_in_sequence(self) -> None:
        """LLM calls two different tools across two iterations."""
        calc_tool = Tool(
            name="calculator",
            description="Math",
            parameters=[ToolParameter(name="expression", type="string", required=True)],
            fn=lambda expression: str(eval(expression)),
        )
        lookup_tool = Tool(
            name="lookup",
            description="Lookup",
            parameters=[ToolParameter(name="key", type="string", required=True)],
            fn=lambda key: f"value_of_{key}",
        )

        round1 = _make_llm_response(
            content="I need to calculate.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "10 * 2"}}],
        )
        round2 = _make_llm_response(
            content="Now let me look something up.",
            tool_calls=[{"name": "lookup", "arguments": {"key": "foo"}}],
        )
        round3 = _make_llm_response(content="The result is 20 and value_of_foo.")
        client = _mock_client([round1, round2, round3])

        agent = AlphaAgent(llm_client=client, tools=[calc_tool, lookup_tool])
        result = agent.analyze("Calculate and look up")

        assert result.iterations_used == 3
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].output == "20"
        assert result.tool_calls[1].output == "value_of_foo"


# ======================================================================
# Tests: error handling
# ======================================================================


class TestErrorHandling:
    """Agent handles tool failures gracefully."""

    def test_tool_exception_captured(self) -> None:
        """A tool that raises returns an error ToolResult and the loop continues."""
        broken = _failing_tool()
        round1 = _make_llm_response(
            content="Let me try the broken tool.",
            tool_calls=[{"name": "broken_tool", "arguments": {}}],
        )
        round2 = _make_llm_response(content="The tool failed, but I'll answer anyway.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[broken])
        result = agent.analyze("Try the broken tool")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].success is False
        assert "deliberate failure" in result.tool_calls[0].error
        # Agent still produced a final answer
        assert "answer anyway" in result.final_answer

    def test_unknown_tool_name(self) -> None:
        """LLM requests a tool that doesn't exist → error captured."""
        round1 = _make_llm_response(
            content="Let me use nonexistent.",
            tool_calls=[{"name": "nonexistent_tool", "arguments": {}}],
        )
        round2 = _make_llm_response(content="OK, moving on.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[])
        result = agent.analyze("Use nonexistent tool")

        assert result.tool_calls[0].success is False
        assert "Unknown tool" in result.tool_calls[0].error

    def test_tool_without_fn(self) -> None:
        """A tool with no callable → error captured."""
        no_fn = _no_fn_tool()
        round1 = _make_llm_response(
            content="Calling no_fn.",
            tool_calls=[{"name": "no_fn", "arguments": {}}],
        )
        round2 = _make_llm_response(content="That didn't work.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[no_fn])
        result = agent.analyze("Call no_fn")

        assert result.tool_calls[0].success is False
        assert "no callable" in result.tool_calls[0].error


# ======================================================================
# Tests: max iterations
# ======================================================================


class TestMaxIterations:
    """Agent respects the iteration budget."""

    def test_loop_terminated_at_max(self) -> None:
        """If the LLM always wants to call a tool, the agent stops at max_iterations."""
        tool = _sample_tool()
        # Simulate an LLM that ALWAYS calls a tool (never gives final answer)
        infinite_call = _make_llm_response(
            content="Let me calculate again.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}}],
        )
        # Repeat 10 times (more than max_iterations)
        client = _mock_client([infinite_call] * 10)

        agent = AlphaAgent(llm_client=client, tools=[tool], max_iterations=3)
        result = agent.analyze("Calculate forever")

        assert result.iterations_used == 3
        assert len(result.tool_calls) == 3
        assert "unable to complete" in result.final_answer.lower()

    def test_max_iterations_of_one(self) -> None:
        """max_iterations=1 → only one LLM call allowed."""
        tool = _sample_tool()
        round1 = _make_llm_response(
            content="I need to calculate.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "5"}}],
        )
        client = _mock_client([round1] * 5)

        agent = AlphaAgent(llm_client=client, tools=[tool], max_iterations=1)
        result = agent.analyze("Calculate")

        # The tool was executed but the budget is spent — no second LLM call
        assert result.iterations_used == 1
        assert len(result.tool_calls) == 1
        # Final answer is the exhaustion message
        assert "unable to complete" in result.final_answer.lower()


# ======================================================================
# Tests: data aggregation
# ======================================================================


class TestDataAggregation:
    """Agent collects structured data from tool outputs."""

    def test_json_tool_output_merged(self) -> None:
        """JSON dict outputs from tools are merged into AgentResponse.data."""
        tool = Tool(
            name="fetch_data",
            description="Fetch data",
            parameters=[],
            fn=lambda: json.dumps({"price": 42.0, "currency": "USD"}),
        )

        round1 = _make_llm_response(
            content="Fetching data.",
            tool_calls=[{"name": "fetch_data", "arguments": {}}],
        )
        round2 = _make_llm_response(content="The price is $42.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[tool])
        result = agent.analyze("What's the price?")

        assert result.data == {"price": 42.0, "currency": "USD"}


class TestExecuteToolSerialization:
    """_execute_tool serializes structured outputs as JSON, not Python repr."""

    def test_dict_output_is_json(self) -> None:
        """A tool returning a dict must produce a JSON-parseable string."""
        tool = Tool(
            name="stock_price",
            description="Get stock price",
            parameters=[ToolParameter(name="symbol", type="string", required=True)],
            fn=lambda symbol: {"symbol": symbol, "price": 319.97, "previous_close": 320.01},
        )
        tc = ToolCall(name="stock_price", arguments={"symbol": "AAPL"})
        agent = AlphaAgent(llm_client=MagicMock(), tools=[tool])

        result = agent._execute_tool(tc)

        assert result.success is True
        # Must be valid JSON (not Python repr with single quotes)
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert parsed["price"] == 319.97
        assert parsed["previous_close"] == 320.01

    def test_list_output_is_json(self) -> None:
        """A tool returning a list must produce a JSON-parseable string."""
        tool = Tool(
            name="multi",
            description="Multi",
            parameters=[],
            fn=lambda: [{"a": 1}, {"b": 2}],
        )
        tc = ToolCall(name="multi", arguments={})
        agent = AlphaAgent(llm_client=MagicMock(), tools=[tool])

        result = agent._execute_tool(tc)
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_string_output_stays_as_str(self) -> None:
        """A tool returning a plain string is NOT JSON-serialized."""
        tool = Tool(
            name="echo",
            description="Echo",
            parameters=[ToolParameter(name="msg", type="string", required=True)],
            fn=lambda msg: f"echo: {msg}",
        )
        tc = ToolCall(name="echo", arguments={"msg": "hello"})
        agent = AlphaAgent(llm_client=MagicMock(), tools=[tool])

        result = agent._execute_tool(tc)
        assert result.output == "echo: hello"


# ======================================================================
# Tests: conversation context
# ======================================================================


class TestConversationContext:
    """Verify messages passed to the LLM are correct."""

    def test_system_prompt_included(self) -> None:
        fake_response = _make_llm_response(content="OK")
        client = _mock_client([fake_response])
        agent = AlphaAgent(llm_client=client)

        agent.analyze("Hello")

        call_args = client.chat_with_tools.call_args
        messages = call_args[0][0]  # first positional arg
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_tool_arguments_sent_as_dict_not_string(self) -> None:
        """Tool-call arguments in the assistant message must be dicts, not JSON strings.

        Ollama returns 400 "Value looks like object, but can't find closing '}'"
        when arguments is a string instead of a dict.
        """
        tool = _sample_tool()
        round1 = _make_llm_response(
            content="Let me calculate.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "2 + 2"}}],
        )
        round2 = _make_llm_response(content="The answer is 4.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[tool])
        agent.analyze("What is 2+2?")

        # The second call re-sends the conversation including the assistant
        # message with tool_calls.  Verify arguments is a dict, NOT a string.
        second_call_args = client.chat_with_tools.call_args_list[1]
        messages = second_call_args[0][0]

        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_msgs) >= 1, "Expected an assistant message with tool_calls"

        tc = assistant_msgs[0]["tool_calls"][0]
        assert isinstance(tc["function"]["arguments"], dict), (
            f"arguments should be a dict, got {type(tc['function']['arguments']).__name__}: "
            f"{tc['function']['arguments']!r}"
        )

    def test_tool_result_appended_to_conversation(self) -> None:
        """After a tool call, the result should appear as a 'tool' message."""
        tool = _sample_tool()
        round1 = _make_llm_response(
            content="Thinking...",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "1"}}],
        )
        round2 = _make_llm_response(content="Done.")
        client = _mock_client([round1, round2])

        agent = AlphaAgent(llm_client=client, tools=[tool])
        agent.analyze("Do math")

        # Second call should include the tool result in messages
        second_call_args = client.chat_with_tools.call_args_list[1]
        messages = second_call_args[0][0]

        # Find tool role messages
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0]["content"] == "1"


# ======================================================================
# Tests: Tool schema conversion
# ======================================================================


class TestToolSchema:
    """Tool.to_ollama_tool() produces valid Ollama tool definitions."""

    def test_basic_conversion(self) -> None:
        tool = _sample_tool()
        ollama_tool = tool.to_ollama_tool()

        assert ollama_tool["type"] == "function"
        assert ollama_tool["function"]["name"] == "calculator"
        assert "expression" in ollama_tool["function"]["parameters"]["properties"]
        assert "expression" in ollama_tool["function"]["parameters"]["required"]

    def test_optional_parameter(self) -> None:
        tool = Tool(
            name="greet",
            description="Greet someone",
            parameters=[
                ToolParameter(name="name", type="string", required=True),
                ToolParameter(name="excited", type="boolean", required=False),
            ],
            fn=lambda name, excited=False: f"Hi {name}!",
        )
        ollama_tool = tool.to_ollama_tool()
        params = ollama_tool["function"]["parameters"]

        assert "name" in params["required"]
        assert "excited" not in params["required"]

    def test_enum_parameter(self) -> None:
        tool = Tool(
            name="set_mode",
            description="Set mode",
            parameters=[
                ToolParameter(name="mode", type="string", enum=["fast", "slow"]),
            ],
            fn=lambda mode: f"Mode set to {mode}",
        )
        ollama_tool = tool.to_ollama_tool()
        mode_schema = ollama_tool["function"]["parameters"]["properties"]["mode"]
        assert mode_schema["enum"] == ["fast", "slow"]


# ======================================================================
# Tests: Tool selection — regression for E2E failures
# ======================================================================


class TestToolSelectionHistory:
    """CASO 1 regression: queries with a time range must use get_stock_history,
    NOT search_news or get_stock_price."""

    def _make_history_tool(self) -> Tool:
        return Tool(
            name="get_stock_history",
            description="Get historical price data over a period",
            parameters=[
                ToolParameter(name="symbol", type="string", required=True),
                ToolParameter(name="period", type="string", required=False),
            ],
            fn=lambda symbol, period="1mo": {
                "symbol": symbol, "period": period, "data_points": 22,
                "high": 190.0, "low": 180.0, "avg": 185.0,
            },
        )

    def _make_price_tool(self) -> Tool:
        return Tool(
            name="get_stock_price",
            description="Get current stock price",
            parameters=[ToolParameter(name="symbol", type="string", required=True)],
            fn=lambda symbol: {"symbol": symbol, "price": 185.5},
        )

    def _make_news_tool(self) -> Tool:
        return Tool(
            name="search_news",
            description="Search financial news",
            parameters=[ToolParameter(name="query", type="string", required=True)],
            fn=lambda query: {"query": query, "count": 0, "results": []},
        )

    def test_history_query_selects_get_stock_history(self) -> None:
        """Agent calls get_stock_history when query mentions 'over the last month'."""
        history_tool = self._make_history_tool()
        price_tool = self._make_price_tool()
        news_tool = self._make_news_tool()

        # LLM round 1: selects get_stock_history (as a well-prompted model should)
        round1 = _make_llm_response(
            content="The query mentions a time range, so I need historical data.",
            tool_calls=[{"name": "get_stock_history", "arguments": {"symbol": "INVALIDTICK123", "period": "1mo"}}],
        )
        # LLM round 2: final answer after seeing error
        round2 = _make_llm_response(
            content="The symbol INVALIDTICK123 returned no historical data."
        )
        client = _mock_client([round1, round2])
        agent = AlphaAgent(
            llm_client=client,
            tools=[history_tool, price_tool, news_tool],
        )

        result = agent.analyze("What is the stock price of INVALIDTICK123 over the last month?")

        # Verify the agent called get_stock_history, NOT search_news or get_stock_price
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_call.name == "get_stock_history"
        assert result.tool_calls[0].tool_call.arguments["symbol"] == "INVALIDTICK123"

    def test_system_prompt_mentions_time_range_for_history(self) -> None:
        """The system prompt must instruct the model to use get_stock_history
        when a time range is present in the query."""
        agent = AlphaAgent(llm_client=MagicMock(), tools=[])
        prompt = agent.system_prompt
        # Key phrases the prompt should contain for CASO 1
        assert "get_stock_history" in prompt
        assert "last month" in prompt.lower() or "last month" in prompt
        assert "time" in prompt.lower()
        # Must explicitly say NOT to use search_news for time-range queries
        assert "search_news" in prompt

    def test_history_tool_description_mentions_time_periods(self) -> None:
        """The get_stock_history tool description should mention time periods
        so the LLM can match period keywords in the query."""
        from alpha_agent.tools import TOOL_REGISTRY

        desc = TOOL_REGISTRY["get_stock_history"].description.lower()
        assert "last month" in desc or "time period" in desc or "time range" in desc
        assert "historical" in desc or "history" in desc

    def test_search_news_description_says_not_fallback(self) -> None:
        """search_news description should say it's NOT a fallback for price queries."""
        from alpha_agent.tools import TOOL_REGISTRY

        desc = TOOL_REGISTRY["search_news"].description.lower()
        assert "fallback" in desc or "do not use" in desc


class TestToolSelectionMissingSymbol:
    """CASO 2 regression: queries without a ticker symbol must result in
    the agent asking for clarification, not calling tools or giving a
    generic answer."""

    def test_agent_asks_for_ticker_when_missing(self) -> None:
        """When the query has no ticker, the LLM should ask the user to specify one."""
        # LLM correctly responds with a clarification request (no tool calls)
        round1 = _make_llm_response(
            content="Could you please specify the ticker symbol for the stock "
                    "you are interested in? For example, AAPL for Apple or MSFT "
                    "for Microsoft."
        )
        client = _mock_client([round1])
        agent = AlphaAgent(llm_client=client, tools=[
            Tool(
                name="get_stock_price",
                description="Get current stock price",
                parameters=[ToolParameter(name="symbol", type="string", required=True)],
                fn=lambda symbol: {"symbol": symbol, "price": 100},
            ),
        ])

        result = agent.analyze("What is the stock price?")

        # No tools should have been called
        assert result.tool_calls == []
        # The answer should contain a request for the ticker symbol
        answer_lower = result.final_answer.lower()
        assert any(
            keyword in answer_lower
            for keyword in ["ticker", "symbol", "specify", "which stock", "which ticker"]
        )

    def test_system_prompt_instructs_missing_symbol_handling(self) -> None:
        """The system prompt must instruct the model to ask for the ticker
        when no symbol is identifiable in the query."""
        agent = AlphaAgent(llm_client=MagicMock(), tools=[])
        prompt = agent.system_prompt
        prompt_lower = prompt.lower()
        # Must mention asking for clarification when symbol is missing
        assert "missing" in prompt_lower or "no" in prompt_lower
        assert "symbol" in prompt_lower or "ticker" in prompt_lower
        assert "ask" in prompt_lower or "specify" in prompt_lower or "clarif" in prompt_lower

    def test_missing_symbol_no_tool_call_with_mock(self) -> None:
        """End-to-end mock: query without symbol → LLM returns no tool calls → agent
        produces a clarification answer (not a generic fallback)."""
        # Even if the LLM returns a final answer without tools, the answer
        # should be a clarification, not a fabricated price or generic text.
        round1 = _make_llm_response(
            content="I need to know which stock you're asking about. "
                    "Please provide the ticker symbol (e.g. AAPL, MSFT, GOOG)."
        )
        client = _mock_client([round1])
        agent = AlphaAgent(llm_client=client, tools=[
            Tool(
                name="get_stock_price",
                description="Get current stock price",
                parameters=[ToolParameter(name="symbol", type="string", required=True)],
                fn=lambda symbol: {"symbol": symbol, "price": 100},
            ),
            Tool(
                name="get_stock_history",
                description="Get historical price data",
                parameters=[ToolParameter(name="symbol", type="string", required=True)],
                fn=lambda symbol, period="1mo": {"symbol": symbol, "period": period},
            ),
        ])

        result = agent.analyze("What is the stock price?")

        assert result.tool_calls == []
        assert "ticker" in result.final_answer.lower() or "symbol" in result.final_answer.lower()


class TestToolSelectionInvalidTicker:
    """CASO 4 regression: the agent must NOT pre-judge a ticker's validity.

    Format-based pre-judgement (e.g. "INVALIDTICK123 is not a valid ticker")
    without calling the tool is wrong: the tool is the source of truth and
    returns a structured ``{"error": ...}`` for unknown symbols.  The agent
    must call the appropriate tool and relay the real error.
    """

    def test_system_prompt_delegates_validation_to_tool(self) -> None:
        agent = AlphaAgent(llm_client=MagicMock(), tools=[])
        prompt = agent.system_prompt.lower()
        # The prompt must tell the model to always call the tool and treat it
        # as the source of truth for ticker validity.
        assert "source of truth" in prompt
        assert "let the tool validate" in prompt
        assert "call" in prompt

    def test_system_prompt_forbids_judging_ticker_from_format(self) -> None:
        agent = AlphaAgent(llm_client=MagicMock(), tools=[])
        prompt = agent.system_prompt.lower()
        # Explicitly forbids format-based pre-judgement of a ticker.
        assert "format is not proof of validity" in prompt
        assert "never judge" in prompt

    def test_system_prompt_instructs_relaying_real_error(self) -> None:
        agent = AlphaAgent(llm_client=MagicMock(), tools=[])
        prompt = agent.system_prompt.lower()
        # When the tool returns an error, the agent must communicate it.
        assert "error" in prompt
        assert "relay" in prompt
        assert "tool reported" in prompt

    def test_agent_calls_history_tool_for_invalid_ticker_and_relays_error(self) -> None:
        """Full mock: for a time-range query with an unknown ticker the agent
        must call get_stock_history (NOT pre-judge the symbol) and relay the
        tool's real error."""
        history_tool = Tool(
            name="get_stock_history",
            description="Get historical price data over a period",
            parameters=[
                ToolParameter(name="symbol", type="string", required=True),
                ToolParameter(name="period", type="string", required=False),
            ],
            fn=lambda symbol, period="1mo": {
                "error": f"No historical data for '{symbol}' with period '{period}'."
            },
        )
        round1 = _make_llm_response(
            content="The query mentions a time range, so I need historical data.",
            tool_calls=[
                {
                    "name": "get_stock_history",
                    "arguments": {"symbol": "INVALIDTICK123", "period": "1mo"},
                }
            ],
        )
        round2 = _make_llm_response(
            content=(
                "get_stock_history returned an error for INVALIDTICK123: "
                "No historical data for 'INVALIDTICK123' with period '1mo'."
            )
        )
        client = _mock_client([round1, round2])
        agent = AlphaAgent(llm_client=client, tools=[history_tool])

        result = agent.analyze("What is the stock price of INVALIDTICK123 over the last month?")

        # The tool WAS called (no format-based pre-judgement) and received the
        # real error from the tool.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_call.name == "get_stock_history"
        assert result.tool_calls[0].tool_call.arguments["symbol"] == "INVALIDTICK123"
        assert result.tool_calls[0].success is True  # the tool answered structurally
        assert "No historical data" in result.tool_calls[0].output
        # The final answer relays the real tool error.
        assert "No historical data" in result.final_answer


# ======================================================================
# Tests: LLMResponse parsing
# ======================================================================


class TestLLMResponseParsing:
    """OllamaClient._parse_response handles various payloads."""

    def test_no_tool_calls(self) -> None:
        data = {"message": {"content": "Hello!", "tool_calls": []}, "done": True}
        resp = OllamaClient._parse_response(data)
        assert resp.content == "Hello!"
        assert resp.tool_calls == []

    def test_with_tool_calls(self) -> None:
        data = {
            "message": {
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test"},
                        }
                    }
                ],
            },
            "done": True,
        }
        resp = OllamaClient._parse_response(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"query": "test"}

    def test_arguments_as_json_string(self) -> None:
        """Some Ollama versions return arguments as a JSON string."""
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": json.dumps({"expr": "1+1"}),
                        }
                    }
                ],
            },
            "done": True,
        }
        resp = OllamaClient._parse_response(data)
        assert resp.tool_calls[0].arguments == {"expr": "1+1"}

    def test_missing_message_key(self) -> None:
        data = {"done": True}
        resp = OllamaClient._parse_response(data)
        assert resp.content == ""
        assert resp.tool_calls == []
