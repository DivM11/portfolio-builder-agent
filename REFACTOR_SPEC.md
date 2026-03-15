# Refactoring Specification: Single-Agent Tool-Based Architecture

## 1. Objective

Replace the multi-LLM chained pipeline (Creator → Evaluator → Orchestrator loop) with a **single reasoning agent** that uses **explicit tools** to accomplish portfolio tasks iteratively. The agent should reason through the problem, invoke tools as needed, and iterate until the user's criteria are met.

### Current State (What We Have)
- **4 separate LLM calls** per iteration, each with distinct models and prompt templates:
  1. Ticker Generation (Creator) — Claude Sonnet
  2. Weight Allocation (Creator) — Gemini Flash
  3. Portfolio Analysis (Evaluator) — Claude Haiku
  4. Portfolio Review/Suggestions (Evaluator) — Claude Sonnet
- **2 agents** (`PortfolioCreatorAgent`, `PortfolioEvaluatorAgent`) coordinated by `AgentOrchestrator`
- Fragile prompt-output parsing (regex for tickers, JSON extraction for weights/suggestions)
- Human-in-the-loop at the end of each iteration (accept/reject suggestions)

### Target State (What We Want)
- **1 reasoning agent** that calls tools to accomplish each sub-task
- The agent decides when to call which tool based on the conversation context
- Iterative refinement happens within the agent's reasoning loop, not via orchestrator state machine
- Human-in-the-loop remains (user approves/rejects suggested changes), but orchestrated by the agent itself
- Clean separation: the LLM reasons, tools execute

---

## 2. Design Decisions

### 2.1 Tool-Based Data Fetching (Not Conditional Python)

> *"Should the call to get financial data be a tool or a python function called conditionally based on the LLM output in previous step?"*

**Decision: Tool.**

Rationale:
- The agent sees tool calls and results in its context, enabling it to reason about data quality, missing tickers, and fetch failures
- Tool-based design lets the agent retry or skip tickers naturally
- Aligns with standard agent patterns (OpenAI function calling, Anthropic tool use)
- Keeps the agent in control of the workflow rather than splitting control flow between Python conditionals and LLM output

### 2.2 Reasoning Model Selection

The main agent should use a **single strong reasoning model** (e.g., Claude Sonnet 4.5 or equivalent) for all steps instead of routing to different models per task. This simplifies the config and improves coherence since one model holds the full context.

The model choice should remain configurable in `config.yml` under a single `agent.model` key.

### 2.3 Iteration Strategy

The agent iterates within a single conversation context:
1. Build initial portfolio → fetch data → analyze → propose
2. User accepts/rejects/modifies → agent refines in-context
3. Loop terminates when user is satisfied or max iterations reached

The agent can see all previous tool results and reasoning, so it doesn't need separate "followup" prompt templates.

---

## 3. Architecture Overview

### 3.1 New Module Structure

```
src/
├── agent.py                    # NEW — Main PortfolioAgent (tool-based)
├── tools/                      # NEW — Tool definitions
│   ├── __init__.py
│   ├── generate_tickers.py     # Tool: Generate ticker recommendations
│   ├── fetch_ticker_data.py    # Tool: Fetch stock data from Massive.com
│   ├── build_summary.py        # Tool: Build portfolio summary from data
│   ├── allocate_weights.py     # Tool: Calculate/normalize portfolio weights
│   └── analyze_portfolio.py    # Tool: Portfolio analysis and risk evaluation
├── config.py                   # MODIFY — Simplify model config
├── dashboard.py                # MODIFY — Replace orchestrator interaction
├── data_client.py              # KEEP — No changes
├── llm_service.py              # MODIFY — Add tool-calling support
├── llm_validation.py           # KEEP — Reuse parsing utilities
├── logging_config.py           # KEEP — No changes
├── plots.py                    # KEEP — No changes
├── portfolio.py                # KEEP — No changes
├── portfolio_display_summary.py # KEEP — No changes
├── summaries.py                # KEEP — No changes
├── tickr_data_manager.py       # KEEP — Used by fetch tool
├── tickr_summary_manager.py    # KEEP — Used by summary tool
├── prompt_validation.py        # REMOVE — Agent does its own validation
├── agent_models.py             # MODIFY — Simplify to single AgentContext
├── agents/                     # REMOVE — Entire directory
│   ├── base.py
│   ├── creator.py
│   ├── evaluator.py
│   └── orchestrator.py
├── event_store/                # KEEP — Update event types for tool calls
│   ├── base.py
│   ├── buffer.py
│   ├── models.py
│   ├── sqlite_store.py
│   └── postgres_store.py
```

### 3.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Dashboard (Streamlit)                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ User chat input → PortfolioAgent.run() → Display result ││
│  └──────────────────────────┬──────────────────────────────┘│
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PortfolioAgent                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              LLM Reasoning Loop                         │ │
│  │  1. Understand user request                             │ │
│  │  2. Call tools as needed (generate_tickers,             │ │
│  │     fetch_data, build_summary, allocate_weights,        │ │
│  │     analyze_portfolio)                                  │ │
│  │  3. Evaluate results, iterate if unsatisfied            │ │
│  │  4. Present final portfolio + analysis to user          │ │
│  └──────┬───────┬───────┬───────┬───────┬─────────────────┘ │
│         │       │       │       │       │                    │
│         ▼       ▼       ▼       ▼       ▼                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Tool Registry                           │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ generate_    │  │ fetch_ticker │  │ build_     │  │   │
│  │  │ tickers      │  │ _data        │  │ summary    │  │   │
│  │  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  │  ┌─────────────┐  ┌──────────────┐                   │   │
│  │  │ allocate_   │  │ analyze_     │                   │   │
│  │  │ weights     │  │ portfolio    │                   │   │
│  │  └─────────────┘  └──────────────┘                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│                     EventStore (instrumentation)             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Massive.com       LLMService      SQLite/PG
        (stock data)      (OpenRouter)    (events)
```

---

## 4. Detailed Tool Specifications

Each tool is a Python function with a JSON schema declaration for the LLM to call. Tools are pure functions that receive structured input and return structured output.

### 4.1 `generate_tickers`

**Purpose:** Recommend US stock ticker symbols based on user preferences.

```python
# Schema for LLM tool calling
{
    "name": "generate_tickers",
    "description": "Generate a list of US stock ticker symbols that match the user's portfolio preferences. Call this for initial portfolio creation or when refining ticker selection.",
    "parameters": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of US stock ticker symbols (e.g., ['AAPL', 'NVDA', 'WMT']). Max 10 tickers."
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why these tickers were selected."
            }
        },
        "required": ["tickers", "reasoning"]
    }
}
```

**Implementation:** This is not an external call — the LLM itself generates the tickers via tool use. The tool handler validates ticker format (uppercase, 1-5 chars) and enforces `max_tickers` from config. Invalid tickers are filtered out and a warning returned.

**Returns:**
```json
{
    "valid_tickers": ["AAPL", "NVDA", "WMT"],
    "rejected_tickers": ["INVALID123"],
    "count": 3
}
```

### 4.2 `fetch_ticker_data`

**Purpose:** Fetch historical price data and financial statements for given tickers from Massive.com.

```python
{
    "name": "fetch_ticker_data",
    "description": "Fetch stock market data (historical prices, financial statements) for the specified ticker symbols. Use this after generating tickers to gather market data.",
    "parameters": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ticker symbols to fetch data for."
            }
        },
        "required": ["tickers"]
    }
}
```

**Implementation:** Wraps `TickrDataManager.fetch_for_tickers()`. Uses existing caching — only fetches tickers not already in cache. Reports per-ticker status.

**Returns:**
```json
{
    "fetched": ["AAPL", "NVDA"],
    "cached": ["WMT"],
    "failed": {"XYZ": "not_found"},
    "available_tickers": ["AAPL", "NVDA", "WMT"]
}
```

### 4.3 `build_summary`

**Purpose:** Build a compact financial summary of fetched ticker data for the agent to reason about.

```python
{
    "name": "build_summary",
    "description": "Build a summary of financial data (price stats, financials) for the given tickers. Call this after fetching ticker data to get a condensed view for analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tickers to include in the summary."
            }
        },
        "required": ["tickers"]
    }
}
```

**Implementation:** Wraps `TickrSummaryManager.build_or_get_summary()`. Uses existing cache.

**Returns:**
```json
{
    "summary": "AAPL: Price(min=142.5, max=198.2, median=172.1, current=195.4), Revenue=94.9B...\nNVDA: ...",
    "ticker_count": 3
}
```

### 4.4 `allocate_weights`

**Purpose:** Apply portfolio weight allocation and normalize to sum to 1.0.

```python
{
    "name": "allocate_weights",
    "description": "Allocate portfolio dollar amounts based on weights and portfolio size. Validates weights sum to ~1.0 and normalizes if needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "weights": {
                "type": "object",
                "description": "Mapping of ticker symbol to weight (0.0-1.0). Must sum to approximately 1.0. Example: {\"AAPL\": 0.5, \"NVDA\": 0.3, \"WMT\": 0.2}"
            },
            "portfolio_size": {
                "type": "number",
                "description": "Total portfolio value in USD."
            }
        },
        "required": ["weights", "portfolio_size"]
    }
}
```

**Implementation:** Reuses `normalize_weights()` and `allocate_portfolio_by_weights()` from `portfolio.py`. Validates weight sum within tolerance.

**Returns:**
```json
{
    "normalized_weights": {"AAPL": 0.5, "NVDA": 0.3, "WMT": 0.2},
    "allocation": {"AAPL": 500.00, "NVDA": 300.00, "WMT": 200.00},
    "portfolio_size": 1000.00
}
```

### 4.5 `analyze_portfolio`

**Purpose:** Run quantitative portfolio analytics (returns, stats, financials).

```python
{
    "name": "analyze_portfolio",
    "description": "Compute portfolio statistics: weighted returns, price stats, financial aggregates. Use this to evaluate portfolio performance after allocating weights.",
    "parameters": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tickers in the portfolio."
            },
            "weights": {
                "type": "object",
                "description": "Weight allocation per ticker."
            }
        },
        "required": ["tickers", "weights"]
    }
}
```

**Implementation:** Wraps `build_portfolio_returns_series()`, `summarize_portfolio_stats()`, `summarize_portfolio_financials()` from `summaries.py`.

**Returns:**
```json
{
    "stats": {"min": 950.2, "max": 1120.5, "median": 1030.1, "current": 1095.4, "return_1y": 0.095},
    "financials": {"Total Revenue": 150000000000, "Net Income": 35000000000},
    "returns_data_points": 252
}
```

---

## 5. Main Agent Design (`src/agent.py`)

### 5.1 System Prompt

The agent receives a system prompt that defines its role, available tools, and behavioral guidelines:

```
You are a portfolio building agent. Your job is to build personalized US equity
portfolios based on user preferences.

## Workflow
1. Understand the user's investment goals from their prompt.
2. Call `generate_tickers` with your recommended stock picks.
3. Call `fetch_ticker_data` to retrieve market data for those tickers.
4. Call `build_summary` to produce a financial overview.
5. Review the summary data. If any tickers look problematic (no data, poor
   fundamentals), revise your picks and repeat steps 2-4.
6. Call `allocate_weights` with your proposed weight distribution.
7. Call `analyze_portfolio` to compute performance metrics.
8. Review the analysis. If the portfolio doesn't meet the user's criteria
   (e.g., too much concentration, poor returns, risk mismatch), adjust
   weights or swap tickers and repeat relevant steps.
9. Present the final portfolio with your analysis.

## Rules
- Maximum {max_tickers} tickers per portfolio.
- Weights must sum to 1.0.
- Never include tickers the user explicitly excluded: {excluded_tickers}
- Always explain your reasoning when presenting results.
- If data fetch fails for a ticker, remove it and pick an alternative.
```

### 5.2 Agent Loop

The `PortfolioAgent` class manages the conversation with the LLM:

```python
class PortfolioAgent:
    """Single reasoning agent that builds portfolios using tools."""

    def __init__(
        self,
        llm_service: LLMService,
        config: dict,
        event_store: EventStore,
        tickr_data_manager: TickrDataManager,
        tickr_summary_manager: TickrSummaryManager,
    ) -> None:
        ...

    def run(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        excluded_tickers: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        """Execute the agent loop until portfolio is complete."""
        ...

    def refine(
        self,
        *,
        feedback: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        """Refine the portfolio based on user feedback."""
        ...
```

### 5.3 Agent Loop Implementation

```
messages = [system_prompt]
messages.append({"role": "user", "content": user_input})

max_tool_rounds = 10  # safety limit

for round in range(max_tool_rounds):
    response = llm_service.complete_with_tools(messages, tools=TOOL_DEFINITIONS)

    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call.name, tool_call.arguments)
            messages.append(tool_call_message)
            messages.append(tool_result_message(result))
            event_store.record(tool_call_event)
    else:
        # Agent has finished reasoning — extract final answer
        break

return parse_agent_result(messages)
```

### 5.4 Conversation Context for Refinement

When the user accepts/rejects/modifies, the agent continues from the existing message history:

```python
def refine(self, *, feedback: str, ...):
    self._messages.append({"role": "user", "content": feedback})
    return self._run_loop()  # same loop, continues with context
```

This replaces the separate `run_initial()` / `run_followup()` / orchestrator state machine.

---

## 6. LLM Service Changes (`src/llm_service.py`)

### 6.1 Add Tool-Calling Support

Add a new method `complete_with_tools()` that passes tool definitions to the OpenAI-compatible API:

```python
def complete_with_tools(
    self,
    messages: list[dict],
    *,
    tools: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> ToolResponse:
    """Call LLM with tool definitions and return response with possible tool calls."""
    response = self.client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Record to event store
    # Parse tool calls or text response
    return ToolResponse(...)
```

### 6.2 New Response Model

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ToolResponse:
    text: str | None
    tool_calls: list[ToolCall]
    has_tool_calls: bool
    raw_response: Any
    usage: dict
```

---

## 7. Config Changes (`config.yml`)

### 7.1 Simplified Model Config

Replace the four per-task model selections with a single agent model:

```yaml
# BEFORE (4 separate models)
openrouter:
  default_models:
    ticker: "anthropic/claude-sonnet-4.5"
    weights: "google/gemini-2.5-flash"
    analysis: "anthropic/claude-haiku-4.5"
    evaluator: "anthropic/claude-sonnet-4.5"

# AFTER (1 agent model)
agent:
  model: "anthropic/claude-sonnet-4.5"
  max_tokens: 4096
  temperature: 0.2
  max_tool_rounds: 10
  max_iterations: 3           # user accept/reject cycles
```

### 7.2 Simplified Prompt Config

Replace the 12+ prompt templates with a single system prompt:

```yaml
# BEFORE (multiple prompt templates)
openrouter:
  prompts:
    ticker_system: "..."
    ticker_template: "..."
    creator_followup_system: "..."
    creator_followup_template: "..."
    weights_system: "..."
    weights_template: "..."
    analysis_system: "..."
    analysis_template: "..."
    evaluator_system: "..."
    evaluator_template: "..."
    evaluator_followup_system: "..."
    evaluator_followup_template: "..."

# AFTER (one system prompt)
agent:
  system_prompt: |
    You are a portfolio building agent. Your job is to build personalized US equity
    portfolios based on user preferences.
    ...
```

### 7.3 Retained Config Sections

These sections remain unchanged:
- `app`, `logging`, `event_store`, `massive`, `dashboard`, `ui`, `stocks`
- `openrouter.api` (base URL, key, etc.)
- `openrouter.model_choices` (for sidebar dropdown)

---

## 8. Dashboard Changes (`src/dashboard.py`)

### 8.1 Replace Orchestrator with Agent

```python
# BEFORE
orchestrator = AgentOrchestrator(creator, evaluator, max_iterations=3, event_store=es)
state = orchestrator.start(user_input=prompt, portfolio_size=size)

# AFTER
agent = PortfolioAgent(llm_service, config, event_store=es, ...)
result = agent.run(user_input=prompt, portfolio_size=size)
```

### 8.2 Simplify Accept/Reject Flow

```python
# BEFORE
if user_accepts:
    state = orchestrator.apply_changes()
elif user_rejects:
    orchestrator.reject_changes()

# AFTER
if user_accepts:
    result = agent.refine(feedback="User accepted the suggested changes. Apply them.")
elif user_rejects:
    # Portfolio is already final, no further action
    pass
```

### 8.3 Sidebar Simplification

Replace 4 model dropdowns with 1:

```python
# BEFORE: 4 dropdowns
st.selectbox("Stock Picker", model_choices)
st.selectbox("Weight Allocator", model_choices)
st.selectbox("Portfolio Analyst", model_choices)
st.selectbox("Portfolio Reviewer", model_choices)

# AFTER: 1 dropdown
st.selectbox("Agent Model", model_choices)
```

---

## 9. Event Store Changes

### 9.1 New Event Types

Add event types for tool-based architecture:

| Event Type | Description |
|---|---|
| `agent_start` | Agent begins processing user request |
| `tool_call` | Agent invokes a tool (name, arguments) |
| `tool_result` | Tool returns result (output, latency) |
| `agent_response` | Agent produces text response |
| `agent_complete` | Agent loop finishes |
| `user_action` | User accept/reject/modify (unchanged) |

### 9.2 EventRecord Additions

Add fields to `EventRecord`:

```python
@dataclass
class EventRecord:
    # ... existing fields ...

    # New fields for tool-based events
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_call_id: str | None = None
    agent_round: int | None = None      # which tool-calling round
```

### 9.3 SQLite Schema Migration

Add new columns to the `events` table. Use `schema_version: 2` in config:

```sql
ALTER TABLE events ADD COLUMN tool_name TEXT;
ALTER TABLE events ADD COLUMN tool_arguments TEXT;   -- JSON
ALTER TABLE events ADD COLUMN tool_result TEXT;       -- JSON
ALTER TABLE events ADD COLUMN tool_call_id TEXT;
ALTER TABLE events ADD COLUMN agent_round INTEGER;
```

---

## 10. AgentResult Simplification (`src/agent_models.py`)

### 10.1 Unified Context

Replace `CreatorContext` + `EvaluatorContext` + `CreatorPrompts` + `EvaluatorPrompts` with:

```python
@dataclass(frozen=True)
class AgentContext:
    user_input: str
    portfolio_size: float
    excluded_tickers: tuple[str, ...] = ()
    session_id: str | None = None
    run_id: str | None = None
```

### 10.2 AgentResult Changes

`AgentResult` from `agents/base.py` moves to `agent_models.py`. Structure is similar but the `metadata` dict captures tool-call history:

```python
@dataclass
class AgentResult:
    tickers: list[str] = field(default_factory=list)
    data_by_ticker: dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    allocation: dict[str, float] = field(default_factory=dict)
    analysis_text: str = ""
    suggestions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)  # full conversation history
```

---

## 11. What Gets Deleted

| File/Module | Reason |
|---|---|
| `src/agents/base.py` | `BaseAgent` abstract class no longer needed (single agent) |
| `src/agents/creator.py` | `PortfolioCreatorAgent` replaced by tools |
| `src/agents/evaluator.py` | `PortfolioEvaluatorAgent` replaced by agent reasoning |
| `src/agents/orchestrator.py` | `AgentOrchestrator` replaced by agent loop |
| `src/agents/__init__.py` | Directory removed |
| `src/prompt_validation.py` | Validation strategies replaced by tool-level validation |

---

## 12. What Gets Kept (Unchanged)

| File/Module | Reason |
|---|---|
| `src/data_client.py` | External API client — tools wrap this |
| `src/portfolio.py` | Allocation math — tools wrap this |
| `src/summaries.py` | Summary builders — tools wrap this |
| `src/plots.py` | Chart rendering — dashboard consumes this |
| `src/portfolio_display_summary.py` | Display formatting — dashboard consumes this |
| `src/tickr_data_manager.py` | Caching layer — tools use this |
| `src/tickr_summary_manager.py` | Caching layer — tools use this |
| `src/logging_config.py` | Logging infra — unchanged |
| `src/llm_validation.py` | Parsing utilities — tools may still use for validation |
| `src/event_store/*` | Event infra — extended, not replaced |

---

## 13. Migration & Testing Strategy

### 13.1 Phase 1 — Foundation (No UI Changes)

1. **Create `src/tools/` directory** with all 5 tool modules.
2. **Write unit tests** for each tool in isolation (test input validation, output format, error cases).
3. **Add `complete_with_tools()`** to `LLMService` and test with mock responses.
4. **Extend `EventRecord`** with tool-call fields; write migration for SQLite schema.
5. **Run existing tests** — nothing should break since no existing code is modified yet.

### 13.2 Phase 2 — Agent Implementation

6. **Create `src/agent.py`** with `PortfolioAgent` class.
7. **Write integration tests** — mock `LLMService` to return scripted tool calls, verify the agent loop produces correct `AgentResult`.
8. **Test `refine()`** — verify conversation context is maintained across refinement cycles.

### 13.3 Phase 3 — Dashboard Integration

9. **Modify `src/dashboard.py`** — swap out orchestrator for agent.
10. **Simplify `config.yml`** — replace multi-model config with single agent config.
11. **Update sidebar** — single model dropdown.
12. **End-to-end test** via Docker: `docker compose run --rm test`.

### 13.4 Phase 4 — Cleanup

13. **Delete `src/agents/` directory** and `src/prompt_validation.py`.
14. **Remove stale config keys** from `config.yml`.
15. **Update `README.md`** with new architecture docs.
16. **Delete old tests**: `test_creator_agent.py`, `test_evaluator_agent.py`, `test_orchestrator.py`, `test_orchestrator_events.py`, `test_prompt_validation.py`.
17. **Write new tests**: `test_agent.py`, `test_tools_*.py`.

### 13.5 Docker Testing at Each Phase

Each phase should pass:
```bash
docker compose run --rm test
```

---

## 14. Roll-Back Plan

- The old `src/agents/` code remains untouched until Phase 4.
- Phases 1-2 are purely additive (new files only).
- Phase 3 is the breaking change — if issues arise, revert `dashboard.py` and `config.yml` to restore the old orchestrator flow.
- Feature flag option: add `agent.use_tool_agent: true/false` in config to toggle between old and new architecture during transition.

---

## 15. Acceptance Criteria

- [ ] Single LLM model handles the entire portfolio workflow via tool calls
- [ ] All 5 tools are implemented with input validation and structured output
- [ ] Agent iterates autonomously — no hardcoded LLM call chain
- [ ] Human-in-the-loop (accept/reject) works via `agent.refine()`
- [ ] Event store captures tool calls with full audit trail
- [ ] SQLite schema migrated to v2 with tool-call columns
- [ ] All existing features work: ticker display, charts, portfolio tab, CSV download
- [ ] Config is simplified to a single model + system prompt
- [ ] Docker tests pass: `docker compose run --rm test`
- [ ] `README.md` updated with new architecture description
- [ ] No regression in existing data fetching, caching, or display logic
