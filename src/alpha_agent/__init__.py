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
    from .cache import ResponseCache  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    from .llm_openai import OpenAIClient  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "AlphaAgent",
    "AntiHallucinationGuardrail",
    "FinancialGuardrail",
    "GuardedAlphaAgent",
    "Guardrail",
    "LLMResponse",
    "OllamaClient",
    "AgentResponse",
    "ResponseCache",
    "OpenAIClient",
    "Tool",
    "ToolCall",
    "ToolParameter",
    "ToolResult",
    "ToolGuardrail",
    "TOOL_REGISTRY",
    "ValidationResult",
]
