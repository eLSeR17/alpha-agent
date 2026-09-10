"""OpenAI-compatible LLM client with function-calling support.

Drop-in replacement for :class:`OllamaClient` that talks to any
OpenAI-compatible API (OpenAI, Azure, Groq, local vLLM, etc.).

The agent only calls ``chat_with_tools()`` — this class implements the
same method and returns the same :class:`LLMResponse` schema, so the
rest of the codebase doesn't need to know which backend is in use.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .schemas import LLMResponse, Tool, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"

try:
    from openai import OpenAI  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment, misc]
    logger.debug("openai package not installed — OpenAIClient unavailable until 'pip install openai'")


class OpenAIClient:
    """Thin wrapper around the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Reads from ``OPENAI_API_KEY`` env var when
        ``None``.
    model:
        Model tag (``gpt-4o-mini``, ``gpt-4o``, ``claude-3-5-sonnet``
        via compatible endpoint, etc.).
    base_url:
        Override for compatible providers (Groq, vLLM, Azure, etc.).
    timeout:
        Per-request timeout in seconds.
    """

    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            )

        kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API — same signature as OllamaClient
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a chat completion request with tool definitions.

        Returns a parsed :class:`LLMResponse` with any tool-calls the
        model requested — identical contract to ``OllamaClient``.
        """
        oa_tools = [self._to_openai_tool(t) for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if oa_tools:
            kwargs["tools"] = oa_tools

        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Simple chat completion *without* tools."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_tool(tool: Tool) -> dict[str, Any]:
        """Convert an internal :class:`Tool` to the OpenAI tools format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in tool.parameters:
            properties[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Extract content and tool-calls from the OpenAI response object."""
        choice = response.choices[0]
        message = choice.message

        content: str = message.content or ""
        parsed_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                func = tc.function
                try:
                    args = json.loads(func.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parsed_calls.append(ToolCall(name=func.name, arguments=args))

        return LLMResponse(
            content=content,
            tool_calls=parsed_calls,
            done=(choice.finish_reason != "length"),
            raw={"finish_reason": choice.finish_reason},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
