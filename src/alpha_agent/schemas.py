"""Pydantic models for AlphaAgent tool definitions, calls, and responses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    """A single parameter in a tool's JSON Schema definition."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    enum: list[str] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema fragment for this parameter."""
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum is not None:
            schema["enum"] = self.enum
        return schema


class Tool(BaseModel):
    """A callable tool the agent can invoke.

    The ``fn`` attribute is a plain Python callable stored at runtime.  It is
    **not** serialised – it lives only in-process.
    """

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    fn: Callable[..., Any] | None = Field(default=None, exclude=True)

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to the ``tools`` format expected by the Ollama /api/chat endpoint."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Tool call / result
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A request from the LLM to call a tool."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Outcome of executing a tool."""

    tool_call: ToolCall
    output: str
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# LLM response (raw)
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Parsed response from the Ollama chat endpoint."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = True
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


# ---------------------------------------------------------------------------
# Agent response (final)
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """What the agent returns to the caller after its ReAct loop."""

    query: str
    final_answer: str
    reasoning: list[str] = Field(default_factory=list)
    tool_calls: list[ToolResult] = Field(default_factory=list)
    iterations_used: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
