"""Unit tests for OpenAIClient — no real network calls.

The OpenAI SDK client is mocked so these tests run offline and fast.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alpha_agent.llm_openai import OpenAIClient
from alpha_agent.schemas import Tool, ToolParameter

# ======================================================================
# Helpers
# ======================================================================


def _make_openai_response(tool_calls: list[dict] | None = None, content: str = "", finish: str = "stop") -> MagicMock:
    """Build a fake OpenAI chat completion response."""
    choice = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None

    if tool_calls:
        calls = []
        for tc in tool_calls:
            call = MagicMock()
            call.function.name = tc["name"]
            call.function.arguments = json.dumps(tc["arguments"])
            calls.append(call)
        msg.tool_calls = calls

    choice.message = msg
    choice.finish_reason = finish
    response = MagicMock()
    response.choices = [choice]
    return response


def _client_with_mock(mock_chat: MagicMock) -> OpenAIClient:
    client = OpenAIClient(api_key="test-key", model="test-model")
    mock_chat.completions.create.return_value = _make_openai_response()
    return client


# ======================================================================
# Client construction
# ======================================================================


class TestConstruction:
    def test_requires_openai_package(self) -> None:
        # Force the module-level import to be None regardless of whether
        # the openai package is installed in the test environment.
        with (
            patch("alpha_agent.llm_openai.OpenAI", None),
            pytest.raises(ImportError),
        ):
            OpenAIClient(api_key="k", model="m")

    def test_builds_client_with_api_key(self) -> None:
        with patch("alpha_agent.llm_openai.OpenAI") as mock_openai:
            OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
            mock_openai.assert_called_once()
            kwargs = mock_openai.call_args.kwargs
            assert kwargs["api_key"] == "sk-test"

    def test_default_model(self) -> None:
        assert OpenAIClient.DEFAULT_MODEL == "gpt-4o-mini"


# ======================================================================
# chat_with_tools
# ======================================================================


class TestChatWithTools:
    def test_parses_tool_calls(self) -> None:
        with patch("alpha_agent.llm_openai.OpenAI") as mock_openai:
            client = OpenAIClient(api_key="k")
            mock_openai.return_value.chat.completions.create.return_value = _make_openai_response(
                tool_calls=[
                    {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
                    {"name": "get_company_info", "arguments": {"symbol": "MSFT"}},
                ]
            )
            resp = client.chat_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )
            assert len(resp.tool_calls) == 2
            assert resp.tool_calls[0].name == "get_stock_price"
            assert resp.tool_calls[0].arguments == {"symbol": "AAPL"}
            assert resp.tool_calls[1].name == "get_company_info"

    def test_parses_malformed_arguments_as_empty(self) -> None:
        with patch("alpha_agent.llm_openai.OpenAI") as mock_openai:
            client = OpenAIClient(api_key="k")
            choice = MagicMock()
            msg = MagicMock()
            msg.content = ""
            call = MagicMock()
            call.function.name = "get_stock_price"
            call.function.arguments = "{not-json"
            msg.tool_calls = [call]
            choice.message = msg
            choice.finish_reason = "stop"
            mock_openai.return_value.chat.completions.create.return_value = MagicMock(choices=[choice])

            resp = client.chat_with_tools([], [])
            assert resp.tool_calls[0].arguments == {}

    def test_empty_content_no_tools(self) -> None:
        with patch("alpha_agent.llm_openai.OpenAI") as mock_openai:
            client = OpenAIClient(api_key="k")
            mock_openai.return_value.chat.completions.create.return_value = _make_openai_response()
            resp = client.chat_with_tools([], [])
            assert resp.content == ""
            assert resp.tool_calls == []
            assert resp.done is True

    def test_finish_reason_length_marks_not_done(self) -> None:
        with patch("alpha_agent.llm_openai.OpenAI") as mock_openai:
            client = OpenAIClient(api_key="k")
            mock_openai.return_value.chat.completions.create.return_value = _make_openai_response(
                content="partial...", finish="length"
            )
            resp = client.chat_with_tools([], [])
            assert resp.done is False

    def test_to_openai_tool_schema(self) -> None:
        tool = Tool(
            name="get_stock_price",
            description="Get price",
            parameters=[
                ToolParameter(name="symbol", type="string", description="Ticker", required=True),
                ToolParameter(name="period", type="string", description="Optional period", required=False),
            ],
        )
        schema = OpenAIClient._to_openai_tool(tool)
        props = schema["function"]["parameters"]["properties"]
        assert list(props.keys()) == ["symbol", "period"]
        assert schema["function"]["parameters"]["required"] == ["symbol"]
