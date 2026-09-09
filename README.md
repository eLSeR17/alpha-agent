# AlphaAgent — Investment Research Agent with Tool Use & Guardrails

[![CI](https://github.com/eLSeR17/alpha-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/eLSeR17/alpha-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **What it demonstrates**: a real agentic system with LLM tool-use (function calling),
> safety guardrails, and grounding against hallucination — built entirely on local
> models (Ollama, no cloud, no API keys).

## Problem

Financial research is time-consuming and error-prone. Analysts must manually gather
data from multiple sources, apply risk calculations, and synthesize news — all while
guarding against AI hallucination and prompt injection. Existing agent frameworks often
lack proper tool-use isolation, making them unsuitable for regulated domains like finance.

## Solution

AlphaAgent is a **ReAct agent** with real function calling that autonomously reasons
about investment queries, selects the appropriate financial tool, and generates
grounded answers. It includes multi-layer guardrails for input validation, tool
rate-limiting, and anti-hallucination anchoring — all running locally on a single
machine with zero cloud dependencies.

## Features (Phase 1)

- **Agent core**: ReAct loop with configurable max iterations (thought → tool decision → execution)
- **Financial tools**: Market data · risk metrics · news · company fundamentals (via yfinance, free)
- **Guardrails**: Prompt-injection detection, financial disclaimer, tool rate-limiting,
  ticker validation, anti-hallucination grounding (numbers anchored to real tool data)
- **191 unit tests**: Agent core, tools, and guardrails
- **E2E validation**: Real Ollama function calling verified end-to-end

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  FinancialGuardrail   │  (injection detection, disclaimer)
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    ToolGuardrail      │  (rate limiting, ticker validation)
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    AlphaAgent (ReAct) │  (thought → decision → action loop)
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    Financial Tools    │  (yfinance: market, risk, news, fundamentals)
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  AntiHallucination    │  (grounding numbers to real tool output)
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    Final Answer       │  (with citations and risk disclaimer)
          └───────────────────────┘
```

## Stack

- **Python 3.11+** — type-hinted, clean, no threads
- **Ollama** (`qwen2.5:7b`, local) — function calling via chat completions API
- **yfinance** — free financial data (market prices, ratios, news)
- **Docker Compose** — `docker_default` network (same pattern as other projects)
- **pytest** — 191 unit tests + E2E validation
- **GitHub Actions** — CI on push/PR (Python 3.11 & 3.12 matrix)

## Getting Started

### Prerequisites
- Ollama running locally with `qwen2.5:7b` pulled (`ollama pull qwen2.5:7b`)
- Python 3.11+ with pip
- (Optional) Docker for containerized execution

### Run tests
```bash
cd alpha-agent
pip install -r requirements.txt
pytest tests/ -v
```

### Run the agent (E2E with real Ollama)

The package uses a `src/` layout, so point `PYTHONPATH` at `src/` and run a small
script that wires the agent with its real financial tools:

```bash
cd alpha-agent
PYTHONPATH=src python - <<'PY'
from alpha_agent import AlphaAgent, GuardedAlphaAgent, OllamaClient
from alpha_agent.tools import TOOL_REGISTRY

llm = OllamaClient()  # connects to local Ollama (qwen2.5:7b)
agent = GuardedAlphaAgent(
    AlphaAgent(llm_client=llm, tools=list(TOOL_REGISTRY.values()))
)
response = agent.analyze("What is the current stock price of AAPL?")
print(response.final_answer)
PY
```

## Tests

**191 unit tests** covering:
- Agent core (ReAct loop, tool selection, iteration limits)
- Financial tools (yfinance wrappers, error handling, rate limiting)
- Guardrails (injection detection, disclaimer enforcement, grounding)
- **Eval harness**: Golden dataset + LLM-as-judge + runner (regression testing)

E2E test validates real function calling with Ollama `qwen2.5:7b` — no mocks.

## CI

The project runs **GitHub Actions CI** on every push and pull request to `main`.
The pipeline tests against **Python 3.11 and 3.12** on `ubuntu-latest`.

All 191 tests are fully deterministic (mocked LLM, mocked yfinance) —
CI requires no external services, no Ollama, and no network access.

## Development process

The full build is documented end-to-end in
[**`docs/DEVELOPMENT_LOG.md`**](docs/DEVELOPMENT_LOG.md): the phase-by-phase
timeline, and six real incident reports (what failed, when, why, how it was fixed,
and how the fix was verified) — from grounding false negatives to a
mis-calibrated LLM-as-judge. It is written to show *how* the agent was made
reliable, not just that it passed: every problem was reproduced against a real
local model, fixed with regression coverage, and re-validated end-to-end
(final real eval run: **mean 0.991 / 9 of 10 cases**).

## Roadmap

- **Persistence/Memory**: Conversation history and context window management
- **Demo**: Live demo + CI pipeline for automated quality checks
- **Evals (shipped)**: Golden dataset + LLM-as-judge + runner
  (see `scripts/run_evals.py` and `tests/test_evals.py`)

## Limitations

- **Not investment advice**: This is a portfolio demonstration, not financial guidance.
  Always consult a qualified professional before making investment decisions.
- **Local model latency**: Ollama `qwen2.5:7b` on CPU adds 5-15s per inference.
  Production deployments would benefit from GPU acceleration or smaller models.
- **yfinance free tier**: Data is subject to rate limits and may lag real-time markets.
  No guarantee of data completeness or accuracy.
- **Educational scope**: Designed to demonstrate agentic patterns, not to replace
  commercial financial analysis tools.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify and distribute with attribution.

---

**Disclaimer**: AlphaAgent is a portfolio project for educational purposes. It does
not provide financial advice, recommendations, or endorsements. All data is for
demonstration only. Past performance does not guarantee future results.

## Related portfolio projects

`alpha-agent` is externally evaluated by
[`evalforge`](https://github.com/eLSeR17/evalforge), a sibling portfolio project
that QAs this agent and
[`smart-contract-rag`](https://github.com/eLSeR17/smart-contract-rag) as
black-box subjects (golden datasets, dual judge, regression guard).