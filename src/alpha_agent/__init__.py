"""AlphaAgent – an AI agent with real function-calling capabilities."""

from .agent import AlphaAgent
from .guarded_agent import GuardedAlphaAgent
from .guardrails import (
    AntiHallucinationGuardrail,
    FinancialGuardrail,
    Guardrail,
    ToolGuardrail,
    ValidationResult,
)
from .llm import OllamaClient
from .schemas import (
    AgentResponse,
    LLMResponse,
    Tool,
    ToolCall,
    ToolParameter,
    ToolResult,
)
from .tools import TOOL_REGISTRY

__all__ = [
    "AlphaAgent",
    "AntiHallucinationGuardrail",
    "FinancialGuardrail",
    "GuardedAlphaAgent",
    "Guardrail",
    "LLMResponse",
    "OllamaClient",
    "AgentResponse",
    "Tool",
    "ToolCall",
    "ToolParameter",
    "ToolResult",
    "ToolGuardrail",
    "TOOL_REGISTRY",
    "ValidationResult",
]
