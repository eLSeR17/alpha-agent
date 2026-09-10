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

# Optional extras — imported lazily so the core stays dependency-light
# (fastapi/openai are only needed when running the API server or the
# OpenAI backend).
try:
    from .cache import ResponseCache
except ImportError:  # pragma: no cover
    pass

try:
    from .llm_openai import OpenAIClient
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "TOOL_REGISTRY",
    "AgentResponse",
    "AlphaAgent",
    "AntiHallucinationGuardrail",
    "FinancialGuardrail",
    "GuardedAlphaAgent",
    "Guardrail",
    "LLMResponse",
    "OllamaClient",
    "OpenAIClient",
    "ResponseCache",
    "Tool",
    "ToolCall",
    "ToolGuardrail",
    "ToolParameter",
    "ToolResult",
    "ValidationResult",
]
