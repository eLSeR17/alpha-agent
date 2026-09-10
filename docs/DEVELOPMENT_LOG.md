# AlphaAgent — Development Log

> A practical, honest account of how AlphaAgent was built: what we planned, what
> broke, when it broke, and exactly how each problem was found and fixed.
> Every incident below is real, reproduced, and verified — nothing is retro-fitted.
> Dates refer to the main build session of **2026-09-08** (single intensive day).

---

## 1. Overview

AlphaAgent is a ReAct (Reason + Act) agent that answers investment research
questions using real financial tools (market data, risk metrics, news, company
fundamentals) and a stack of guardrails (prompt-injection detection, tool
rate-limiting, ticker validation, anti-hallucination grounding). It runs fully
locally on a 7B parameter model (Ollama), with zero cloud dependencies.

| Phase | Goal | Deliverables | Status |
|-------|------|--------------|--------|
| **Phase 1** | Agent core with tool use and safety guardrails | `agent.py` (ReAct loop), `llm.py` (`OllamaClient`, qwen2.5:7b via `http://ollama:11434` on the Docker network), `schemas.py`, guardrails (`financial`, `tool_guard`, `anti_hallucination`), 5 financial tools on yfinance (`get_stock_price`, `get_stock_history`, `calculate_risk_metrics`, `search_news`, `get_company_info`) | ✅ Completed |
| **Phase 2** | Eval harness for regression testing with real LLM judging | Golden set (10 cases: price, multiple-tools, failure-mode, edge-case, malicious), runner, LLM-as-judge + `MockJudge`, report generator, `scripts/run_evals.py` | ✅ Completed |
| **Phase 3** | CI so the whole suite runs on every push/PR | GitHub Actions workflow (Python 3.11 & 3.12), `pytest.ini`, CI badge | ✅ Completed |

**Closing state**: 191 unit tests, all deterministic (LLM and yfinance fully
mocked — CI needs no Ollama, no network), plus E2E validation against a real
local model. Final real E2E eval run: **mean score 0.991 (9/10 cases)**.

---

## 2. Development timeline

| Date | Phase | Milestone | Status |
|------|-------|-----------|--------|
| 2026-09-08 | 1 | Agent skeleton: schemas, OllamaClient, ReAct loop landed | ✅ |
| 2026-09-08 | 1 | Five yfinance tools + tool registry + guardrails (financial, tool guard, anti-hallucination) | ✅ |
| 2026-09-08 | 1 | First real E2E run against Ollama → **INC-001** (grounding false negatives) | ✅ fixed |
| 2026-09-08 | 1 | 140+ Phase-1 unit tests green, all deterministic | ✅ |
| 2026-09-08 | 2 | Golden set (10 cases), runner, MockJudge, LLM-as-judge, report | ✅ |
| 2026-09-08 | 2 | First full E2E eval run → **INC-002** (judge outputs all zeros) | ✅ fixed |
| 2026-09-08 | 2 | Eval run after parse fix: mean 0.799 (8/10), tool-selection defects visible | ✅ diagnosed |
| 2026-09-08 | 2 | **INC-003** (wrong tool on failure-mode/edge-case) fixed via prompt rules | ✅ fixed |
| 2026-09-08 | 2 | **INC-004** (over-correction + bad golden case design) fixed; mean ~0.826 | ✅ fixed |
| 2026-09-08 | 2 | **INC-005** (judge miscalibrated for error/clarification answers) fixed | ✅ fixed |
| 2026-09-08 | 2 | Final real E2E: **mean 0.991, 9/10** — remaining miss documented honestly | ✅ |
| 2026-09-08 | 3 | CI workflow (3.11/3.12 matrix) + badge; full suite runs headless | ✅ |
| 2026-09-08 | 3 | Review pass → **INC-006** (process incident): false-positive REJECT + 2 real findings fixed | ✅ |
| 2026-09-10 | 4 | **HTTP API server** (FastAPI): `/ask`, `/health`, `/tools` endpoints | ✅ |
| 2026-09-10 | 4 | **OpenAIClient** — drop-in OpenAI-compatible backend (gpt-4o-mini, Groq, vLLM) | ✅ |
| 2026-09-10 | 4 | **ResponseCache** — SQLite query cache (TTL-configurable, optional) | ✅ |
| 2026-09-10 | 4 | API hardening: 503 on backend-unavailable, sandboxed 422 on bad input | ✅ |

---

## 3. Incident reports

### INC-001 — AntiHallucinationGuardrail false negative: grounding could not see real tool numbers

- **When**: 2026-09-08, Phase 1 — detected during the **first real E2E run** against Ollama.
- **Symptom**: answers that contained *real* numbers returned by the tools
  (e.g. `AAPL 319.97`) came back with the annotation
  `"⚠️ could not be verified: 319.97, 320.01"` and the guardrail flagged the
  response as `allowed=False` — i.e. the output was **blocked even though every
  number was factual**.
- **Impact**: the anti-hallucination layer — the most safety-critical component —
  produced false positives on correct answers, degrading every E2E response and
  risking a "boy who cried wolf" effect.
- **Root cause**: in `agent.py::_execute_tool` the tool output was serialized with
  `str(output)`, producing a Python `repr` (`{'symbol': 'AAPL', 'price': 319.97}`,
  single quotes) instead of valid JSON. Downstream,
  `guarded_agent._extract_tool_data` did `json.loads(...)` on that text → raised →
  swallowed → `tool_data` came back **empty**, so there was no reference data to
  verify numbers against.
- **Fix**: two-sided, defense in depth:
  1. `agent.py` now serializes structured tool output with
     `json.dumps(output, default=str)` (dicts/lists become valid JSON; strings stay
     as-is).
  2. `guarded_agent._extract_tool_data` gained a fallback parser using
     `ast.literal_eval` so legacy `repr`-shaped text still parses.
- **Verification**: real E2E run with Ollama → `allowed=True`,
  `response_numbers_checked: 2`, `tool_numbers_available: 3`, clean final answer
  with no "could not be verified" note. Regression tests added for
  dict/list serialization and the `repr` fallback path.
- **Lesson**: never serialize structured tool output with `str()` when a downstream
  consumer will parse it as JSON — normalize the format at the source *and* add
  tolerance at the parser.

---

### INC-002 — LLM-as-judge "non-JSON": every case scored 0.000

- **When**: 2026-09-08, Phase 2 — during the **first full E2E eval run**.
- **Symptom**: `run_evals.py` reported a mean score of `0.000` with all
  non-malicious cases at zero, and the warning
  `"Judge LLM returned non-JSON; scoring all zeros"`.
- **Impact**: the eval harness was useless — it could not distinguish a good agent
  from a broken one, so no improvement could be trusted.
- **Root cause**: the orchestrator instrumented `_parse_score` and captured the
  *raw* judge content:
  `'{{"grounded": 0.5, "correct_tool": 1.0, "clarity": 0.8, "relevance": 0.0}}'`.
  The model sometimes wraps the JSON in **double braces** (`{{...}}`), which is not
  valid JSON and was covered by none of the existing cascade steps (direct parse,
  brace matching, fenced blocks).
- **Fix**: added a "peel one outer brace" attempt to `_parse_score` — when direct
  and brace-matched parsing fail, strip one leading `{` and one trailing `}` and
  retry — inserted between the brace-match step and the fences step.
- **Verification**: regression tests with the exact captured input
  (`test_parse_double_brace_wrapped`, `test_parse_double_brace_with_surrounding_text`);
  then a real E2E run where `grounded`, `correct_tool`, `clarity` and `relevance`
  came back non-zero.
- **Lesson**: LLM-as-judge output is unpredictable by nature — the parser must be a
  tolerant cascade — and debugging must look at the raw content, not only at the
  parsed result.

---

### INC-003 — Evals exposed tool-selection failures (wrong tool, missing prompt for missing symbol)

- **When**: 2026-09-08, Phase 2 — real E2E run after the INC-002 parse fix
  (mean 0.799, 8/10 — the harness was finally measuring truthfully).
- **Symptom**:
  - failure-mode case *"INVALIDTICK123 over the last month"* called `search_news`
    instead of `get_stock_history`.
  - edge-case *"What is the stock price?"* (no symbol) called **no tool** but
    answered with a generic *"Based on the available information..."* instead of
    asking for the ticker.
- **Impact**: real-user scenarios handled incorrectly — a user with a bad ticker
  got news instead of an error, and a query without a symbol got a vague answer
  instead of a clarifying question.
- **Root cause**: the `SYSTEM_PROMPT` had no explicit tool-selection rules: nothing
  mapped a time range to `get_stock_history`, and nothing said "no symbol → ask,
  don't guess".
- **Fix**: rewrote the `SYSTEM_PROMPT` with an explicit **"Tool selection rules"**
  section (time range → `get_stock_history`; `search_news` is *not* a fallback for
  history) and a **"Missing symbol handling"** section (no ticker → do not call
  tools, ask the user for the ticker). Updated the tool descriptions accordingly
  (`get_stock_history` now mentions time periods; `search_news` explicitly not a
  fallback).
- **Verification**: new prompt/mock tests (`test_history_query_selects_get_stock_history`,
  `test_agent_asks_for_ticker_when_missing`, `test_search_news_description_says_not_fallback`,
  …) plus a subsequent real E2E run confirming the corrected behavior.
- **Lesson**: with open-source local models, tool selection must be guided
  explicitly in the prompt — and it is the *real* E2E evals that expose these
  failures; mocked tests alone never will.

---

### INC-004 — The INC-003 fix over-corrected: the agent pre-judged tickers, and the golden case rewarded the wrong behavior

- **When**: 2026-09-08, Phase 2 — real E2E run immediately after INC-003.
- **Symptom**:
  - failure-mode case: the agent called **no tool** at all and answered
    *"The ticker symbol \"INVALIDTICK123\" does not appear to be a valid stock ticker"* —
    it pre-judged the symbol's validity from its format instead of letting the
    tool return the real error.
  - edge-case case: the agent now correctly asked for the ticker (desired
    behavior!), but the eval marked it **FAIL** because the golden case required
    `expected_tool='get_stock_price'` — i.e. the dataset rewarded calling a tool
    when there was no symbol to look up.
- **Impact**: two coupled mistakes — one in the agent (format-based pre-judging,
  which would also wrongly *accept* fake-but-well-formed tickers) and one in the
  test design (the eval punished the correct clarification behavior).
- **Root cause**: (a) the new prompt rules led the model to auto-validate ticker
  *format* instead of delegating validation to the tool; (b) the edge-case golden
  entry was designed before we had defined the intended behavior for
  symbol-less queries.
- **Fix**:
  - **Prompt**: added *"Never judge a ticker's validity yourself; string format is
    not proof of validity. The tool is the source of truth and returns a structured
    `{"error": ...}` — call it and relay its error."*
  - **Golden set**: redesigned the edge-case entry to `expected_tool: null` with
    the semantic "no tool should be called — ask for the ticker", plus
    `min_evidence: ["ticker", "symbol"]`. Aligned the runner and `MockJudge` so
    `expected_tool=None` requires **zero** tool calls.
- **Verification**: scripted simulation + real E2E run — mean improved (~0.826),
  cases no longer failed with "tool not called", and the edge case passed only when
  the agent asked for the ticker without calling any tool.
- **Lesson**: when tuning an agent, always re-check whether the eval case encodes
  the behavior you actually want (an eval that rewards calling a tool with no data
  is broken) — and the *tool*, not the LLM, must be the source of truth for input
  validity.

---

### INC-005 — LLM-as-judge miscalibrated for error/clarification answers: honest correct responses scored zero

- **When**: 2026-09-08, Phase 2 — real E2E run after INC-004.
- **Symptom**: cases 4 (failure-mode) and 7 (edge-case) now passed all
  deterministic checks (correct tool called / zero tools + ticker requested), but
  the judge scored them `grounded: 0.0, correct_tool: 0.0, relevance: 0.0` and they
  failed solely on "judge score below threshold".
- **Impact**: genuinely correct, honest agent behavior was being penalized — the
  eval would keep "fixing" things that were already right, or worse, push the agent
  toward inventing numbers just to satisfy the judge.
- **Root cause**: the judge prompt assumed every answer must contain verifiable
  numbers and at least one tool call. For honest error/clarification responses —
  where *not* calling a tool is the right answer — it zeroed everything.
- **Fix**: pass `category` through the runner into the `JudgeContext`, and made the
  judge prompt **category-aware**:
  - `failure-mode` → `grounded` is HIGH when the answer is honest and invents no
    numbers (LOW when it fabricates);
  - `edge-case` → `correct_tool` is HIGH when "not calling a tool" was the correct
    behavior.
  The `MockJudge` was updated to mirror the same rules.
- **Verification**: new calibration tests (`test_mock_judge_failure_mode_honest_error`,
  `test_prompt_includes_failure_mode_rules`, …). Final real E2E run:
  **mean 0.991, 9/10** (see Section 6 for the honest breakdown of the one miss).
- **Lesson**: a generic LLM-as-judge does not fit every case type — give the judge
  the case context (`category`) and explicit rules to distinguish an honest
  "no-data" answer from fabrication.

---

### INC-006 — Review pass: false-positive REJECT surfaced two real repo issues (process incident)

- **When**: 2026-09-08, Phase 3 — final review of the repo before publication.
- **Symptom**: the pipeline's automated review step issued a **REJECT** with reason
  **B1**: *"the `golden_set.json` file is ignored by `.gitignore` (`data/*.json`)"*.
- **Impact**: a blocking review verdict based on an incorrect premise; had it been
  accepted blindly, time would have been wasted "un-ignoring" a file that was
  never ignored.
- **Root cause of the false positive**: the `.gitignore` glob `data/*.json` was
  misread by the reviewer — that pattern does **not** cross subdirectories, so it
  does not affect `data/golden/golden_set.json`. Verified independently with
  `git check-ignore`, which confirmed the golden file is **not** ignored.
- **Real findings the same REJECT uncovered**:
  - **M1**: `README.md` referenced `src/alpha_agent/cli.py`, a file that does not
    exist → replaced with a working, verified snippet that wires the agent with its
    real tools.
  - **M3**: generated eval artifacts (`report_*.json` produced by `run_evals.py`)
    were **not** excluded from git → added `**/data/evals/report_*.json` to
    `.gitignore` and hardened the golden set with an explicit negation
    (`!data/golden/*.json`).
- **Verification**: `git check-ignore` output for both the golden file (not ignored)
  and the report artifacts (now ignored); README snippet executed successfully
  against the real stack (E2E).
- **Lesson**: verify independent agents' PASS/FAIL verdicts against raw evidence —
  this REJECT was one false positive and two genuine catches. The review pipeline
  works; the reviewer just needs the same evidence-based scrutiny as everyone else.

---

## 4. Security & privacy notes

| Item | Decision | Why |
|------|----------|-----|
| **Git author identity** | `eLSeR17 <112463505+eLSeR17@users.noreply.github.com>` | The repository is public. Using a `noreply` address keeps the personal e-mail off GitHub while commits still link to the profile. |
| **Secrets** | None committed — `.env` only via `.env.example` with placeholders; `.gitignore` excludes `.env`/`.env.local` | Public repo rule: never ship keys, tokens or credentials. This project needs no API keys at all (local model + free yfinance data). |
| **Confidentiality** | No internal infrastructure details (host paths, internal service names, private ports) appear in repo content | The repo is self-contained: it documents only its own stack (local Ollama container, Docker network, yfinance) so anyone can reproduce it. |
| **Local-only LLM** | All inference via local Ollama (`qwen2.5:7b`), no cloud provider, no API keys | Cheaper, private, and makes CI reproducible without external model access. |

Prompt-injection protection is also a *product* feature, not just a repo policy:
the agent's `financial` guardrail blocks jailbreak/impersonation patterns before
any tool runs, and the eval suite carries two dedicated `malicious` golden cases
to verify the block.

---

## 5. Testing & validation summary

| Suite | Tests | What it covers |
|-------|-------|----------------|
| `tests/test_agent.py` | ~45 | ReAct loop, tool selection, error handling, iteration limits, output serialization (INC-001 regression), prompt rules (INC-003/INC-004 regressions) |
| `tests/test_tools.py` | ~40 | Each of the 5 financial tools: happy paths, error handling, rate limits, registry integrity |
| `tests/test_guardrails.py` | ~65 | Injection detection, disclaimer enforcement, ticker validation, rate limiting, anti-hallucination grounding + integration |
| `tests/test_evals.py` | ~38 | Golden-set loader, runner semantics (tool/evidence checks), MockJudge, judge parse cascade (INC-002 regression), category-aware calibration (INC-005 regression) |
| `tests/test_integration.py` | 5 | Default model and base URL wiring (Docker-network Ollama) |
| **Total** | **191** | All deterministic: LLM and yfinance fully mocked, no network, no Ollama → safe for CI |

**Real E2E validation** (no mocks) against local Ollama `qwen2.5:7b`, run after
each fix:

| Run | Mean score | Detail |
|-----|-----------|--------|
| 1st full eval (INC-002 active) | 0.000 | Judge parser broken (double-brace JSON) — harness blind |
| After INC-002 parse fix | 0.799 (8/10) | Harness truthful; tool-selection defects visible |
| After INC-003 | — | Behavioral fix verified on failure-mode/edge-case cases |
| After INC-004 | ~0.826 | Agent no longer pre-judges tickers; edge-case passes with zero tools |
| **Final (after INC-005)** | **0.991 (9/10)** | Judge calibrated per category; only real-model variance remains |

### Why 9/10 and not 10/10

The single remaining miss is a genuinely useful signal, not noise: on one
multiple-tools case the local 7B model picked `get_stock_price` where
`get_stock_history` was expected (same ticker present in both results, so the
answer remained factually correct). This is inherent variance in a small
open-source model's tool selection. The eval *detecting* it — deterministically,
case by case — is evidence the harness works, not that it is broken. We prefer an
honest 9/10 with full visibility over a gamed 10/10.

---

## 5b. Phase 4 — HTTP API server, dual backend, response cache (2026-09-10)

The agent core was already production-shaped (guardrails, grounding, eval
harness). Phase 4 turns it into a **service** so it can be integrated by real
clients (web UI, Slack bot, another service):

- **`api.py` (FastAPI)**: `POST /ask` (guarded agent), `GET /health`
  (backend + model + cache status), `GET /tools` (registry introspection).
  Validation via Pydantic: empty/blank questions → 422, oversized → 422.
  Backend-unavailable (Ollama down, network error) → **503 with a
  self-diagnosing message**, never a bare 500.
- **`llm_openai.py` (`OpenAIClient`)**: implements the exact same
  `chat_with_tools()` contract as `OllamaClient` so `AlphaAgent` is backend
  agnostic. Supports any OpenAI-compatible endpoint (OpenAI, Groq, Azure,
  vLLM). Selection via one env var: `ALPHA_AGENT_BACKEND=ollama|openai`.
  Parsing is tolerant: malformed `arguments` JSON → `{}`, never a crash.
- **`cache.py` (`ResponseCache`)**: SQLite cache keyed by SHA-256 of the
  query, TTL-configurable (default 1h), thread-safe. Repeated identical
  questions are answered from cache (saves tokens on paid backends).
  Serialization round-trips the full `AgentResponse` (tool calls included).
- **Tests**: +18 deterministic tests (OpenAI client parsing/schema, cache
  hit/miss/TTL/round-trip, API endpoints with scripted fake agent). Suite
  total: **209**, all hermetic, CI-safe.

Design notes (why it's shaped this way):

- **Backend swap is one env var, not a code change** — the agent core only
  knows `chat_with_tools()`. This keeps the public repo cloud-free by
  default while making the paid path an opt-in.
- **Caching is opt-in** (`ALPHA_AGENT_CACHE=1`) — no behaviour change for
  existing users; `from_cache: true` in the response makes hits visible.
- **503, not 500** — a service must distinguish "I'm broken" from "my
  dependency is down". The message tells the operator exactly which env var
  to check. Verified live with Ollama unreachable.

## 6. How to reproduce the key checks

```bash
# Full deterministic suite (no Ollama, no network needed)
pip install -r requirements.txt
pytest tests/ -v

# Real E2E evals against local Ollama (requires Docker network + qwen2.5:7b pulled)
python scripts/run_evals.py
```

See the root `README.md` for architecture, getting started, and limitations.