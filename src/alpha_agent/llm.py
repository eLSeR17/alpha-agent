"""Ollama LLM client with native function-calling support."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .schemas import LLMResponse, Tool, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://ollama:11434"
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaClient:
    """Thin HTTP wrapper around the Ollama ``/api/chat`` endpoint.

    Parameters
    ----------
    base_url:
        Reachable address for Ollama **inside the Docker network**.
        Never use ``localhost`` – it won't resolve from a container.
    model:
        Ollama model tag to use for completions.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a chat completion request with tool definitions.

        Returns a parsed :class:`LLMResponse` with any tool-calls the model
        requested.
        """
        ollama_tools = [t.to_ollama_tool() for t in tools] if tools else []

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if ollama_tools:
            body["tools"] = ollama_tools

        resp = self._http.post("/api/chat", json=body)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        return self._parse_response(data)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Simple chat completion *without* tools."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = self._http.post("/api/chat", json=body)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return self._parse_response(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Extract content and tool-calls from the Ollama response payload."""
        message = data.get("message", {})
        content: str = message.get("content", "") or ""
        raw_tool_calls: list[dict[str, Any]] = message.get("tool_calls", [])

        parsed_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            # Ollama may return arguments as a JSON string or a dict
            args_raw = func.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw
            parsed_calls.append(ToolCall(name=name, arguments=args))

        return LLMResponse(
            content=content,
            tool_calls=parsed_calls,
            done=data.get("done", True),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
