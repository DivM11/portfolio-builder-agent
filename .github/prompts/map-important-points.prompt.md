---
description: "Remember all the important points and their functionality in the codebase: config.yml, Dockerfile, docker-compose.yml, pyproject.toml, src modules, and their tests"
name: "Remember Codebase"
argument-hint: "Optional focus area, e.g. \"monitoring stack\" or \"agent loop\""
agent: "agent"
---
Read the files listed below and produce a complete reference of every important point and its functionality.
The output must be concrete enough that an agent reading it later can understand the full codebase without re-reading files.
If arguments were provided with this prompt, narrow or deepen focus accordingly — but still cover all listed files at minimum.

## Files to read

Read each of these files before producing the output:

### Infrastructure & config
- [config.yml](../../config.yml) — all runtime configuration (app, logging, event_store, massive API, dashboard, UI text, agent settings, openrouter, stocks).
- [Dockerfile](../../Dockerfile) — container build: base image, Poetry install, copy, exposed ports, default CMD.
- [docker-compose.yml](../../docker-compose.yml) — services (app :8501, monitor-api :8000, monitor-ui :8502, test, event-db), shared `event-data` volume, profiles.
- [pyproject.toml](../../pyproject.toml) — Poetry deps (streamlit, massive, openai, pandas, plotly, fastapi, …), dev deps (ruff, mypy, pytest), tool configs (ruff rules, mypy strictness, coverage ≥90%).

### Source — top-level modules (`src/`)
- [src/config.py](../../src/config.py) — `load_config()`: merges config.yml with env-var API keys.
- [src/agent.py](../../src/agent.py) — `PortfolioAgent`: iterative tool-loop that calls five tools in sequence; emits `ToolCallRecord`.
- [src/agent_models.py](../../src/agent_models.py) — `AgentContext`, `AgentResult`, `Context`: data models for agent state and results.
- [src/llm_service.py](../../src/llm_service.py) — `LLMService`: OpenRouter/OpenAI chat completion wrapper; emits `LLMCallRecord`.
- [src/llm_validation.py](../../src/llm_validation.py) — `extract_valid_tickers()`, `parse_weights_payload()`, `validate_weight_sum()`: LLM output parsing.
- [src/input_guard.py](../../src/input_guard.py) — `InputGuard`: LLM-based prompt-injection and off-topic screening.
- [src/data_client.py](../../src/data_client.py) — `create_massive_client()`, `fetch_price_history()`, `fetch_stock_data()`: Massive.com API wrapper.
- [src/dashboard.py](../../src/dashboard.py) — `run_dashboard()`: Streamlit UI with chat, historical prices, and portfolio tabs.
- [src/portfolio.py](../../src/portfolio.py) — `normalize_weights()`, `allocate_portfolio()`, `allocate_portfolio_by_weights()`: weight normalization and dollar allocation.
- [src/portfolio_display_summary.py](../../src/portfolio_display_summary.py) — `PortfolioDisplaySummary`: formats portfolio headers and suggested changes for display.
- [src/summaries.py](../../src/summaries.py) — `build_ticker_summary()`, `build_portfolio_summary()`, `build_portfolio_returns_series()`: data summarization.
- [src/plots.py](../../src/plots.py) — `plot_history()`, `plot_portfolio_allocation()`, `plot_portfolio_returns()`: Plotly charts.
- [src/tickr_data_manager.py](../../src/tickr_data_manager.py) — `TickrDataManager`: per-ticker caching fetch manager.
- [src/tickr_summary_manager.py](../../src/tickr_summary_manager.py) — `TickrSummaryManager`: caches summaries keyed by ticker set + cache version.
- [src/monitoring_api.py](../../src/monitoring_api.py) — FastAPI app with `/health`, `/events`, `/llm-calls`, `/tool-calls`, `/agent-performance` endpoints.
- [src/logging_config.py](../../src/logging_config.py) — `configure_logging()`, `JsonFormatter`, `CorrelationFilter`: structured JSON logging.

### Source — sub-packages
- [src/tools/](../../src/tools/) — Five agent tools, each exporting `*_tool()` + `tool_definition()`:
  - `generate_tickers` — validates/normalises candidate ticker symbols.
  - `fetch_ticker_data` — fetches historical price + financial data per ticker.
  - `build_summary` — compact JSON stats summary for a ticker set.
  - `allocate_weights` — normalises weights and converts to dollar allocation.
  - `analyze_portfolio` — computes performance metrics and financial aggregates.
- [src/event_store/](../../src/event_store/) — Pluggable event persistence:
  - `base.py` — `EventStore` / `MonitoringStore` protocols + `NullEventStore`.
  - `models.py` — `EventRecord`, `LLMCallRecord`, `ToolCallRecord`, `AgentPerformanceRecord`.
  - `sqlite_store.py` — `SQLiteEventStore`: four-table SQLite backend.
  - `buffer.py` — `BufferedEventStore`: batched-write wrapper.
  - `postgres_store.py` — `PostgresEventStore`: stub (raises NotImplementedError).
  - `__init__.py` — `create_event_store()` factory.
- [src/etl/](../../src/etl/) — `agent_performance.py` → `materialise_agent_performance()`: aggregates LLM + tool records into per-run summary.

### Tests (`tests/`)
Each test file maps to a source module. Read these to understand coverage and test patterns:
- [tests/test_agent.py](../../tests/test_agent.py) — `PortfolioAgent` with mocked LLM (`CaptureEventStore`, `CaptureLLMService`).
- [tests/test_allocate_weights.py](../../tests/test_allocate_weights.py) — weight normalisation + schema.
- [tests/test_config.py](../../tests/test_config.py) — YAML loading, env-var injection.
- [tests/test_context.py](../../tests/test_context.py) — `Context` message/state management.
- [tests/test_dashboard.py](../../tests/test_dashboard.py) — `run_dashboard()` with dummy Streamlit.
- [tests/test_data_client.py](../../tests/test_data_client.py) — Massive.com client with stubbed API.
- [tests/test_etl_agent_performance.py](../../tests/test_etl_agent_performance.py) — ETL materialisation.
- [tests/test_event_store.py](../../tests/test_event_store.py) — factory, SQLite roundtrip, backend selection.
- [tests/test_fetch_ticker_data.py](../../tests/test_fetch_ticker_data.py) — fetch tool with mock `TickrDataManager`.
- [tests/test_input_guard.py](../../tests/test_input_guard.py) — `InputGuard` with stub LLM.
- [tests/test_llm_service.py](../../tests/test_llm_service.py) — `LLMService` with mocked OpenAI client.
- [tests/test_llm_service_events.py](../../tests/test_llm_service_events.py) — LLM event recording.
- [tests/test_llm_validation.py](../../tests/test_llm_validation.py) — ticker/weight/suggestion parsing.
- [tests/test_logging_config.py](../../tests/test_logging_config.py) — JSON logging, correlation IDs.
- [tests/test_main.py](../../tests/test_main.py) — `main.py` entrypoint.
- [tests/test_monitoring_api.py](../../tests/test_monitoring_api.py) — FastAPI endpoint queries.
- [tests/test_monitoring_store.py](../../tests/test_monitoring_store.py) — `MonitoringStore` on SQLite + Null.
- [tests/test_plots.py](../../tests/test_plots.py) — Plotly chart builders.
- [tests/test_portfolio.py](../../tests/test_portfolio.py) — allocation, normalisation, Decimal precision.
- [tests/test_portfolio_display_summary.py](../../tests/test_portfolio_display_summary.py) — display formatting.
- [tests/test_summaries.py](../../tests/test_summaries.py) — ticker/portfolio summarisation.
- [tests/test_tickr_data_manager.py](../../tests/test_tickr_data_manager.py) — caching + incremental fetch.
- [tests/test_tickr_summary_manager.py](../../tests/test_tickr_summary_manager.py) — version-keyed summary cache.

Also read [README.md](../../README.md) for architecture context, the system diagram, and agent design overview.

## Output format

Produce exactly these sections:

### 1. Infrastructure & Config Points
For each of `config.yml`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml`:
- List every important setting/block and what it controls.
- Note which source modules consume each setting.

### 2. Source Module Map
For each module under `src/`:
- Key classes/functions and their responsibility.
- Which other modules it depends on and which depend on it.
- Important constants, patterns, or non-obvious behavior.

### 3. Agent Tool Chain
- The five tools in execution order, with inputs/outputs of each.
- How the iterative loop decides to continue or stop.
- Caching layers (`TickrDataManager`, `TickrSummaryManager`) and when caches invalidate.

### 4. Event & Monitoring Pipeline
- Four event store tables, what writes to each, and what each row represents.
- How the monitoring API and dashboard consume the data.
- ETL aggregation flow.

### 5. Test Coverage Map
Table with columns: Source module | Test file | What is tested | Gaps or notes.

### 6. Cross-Cutting Concerns
- Logging strategy and correlation IDs.
- Input guard / security boundary.
- Error handling and fallback patterns.
- Docker volume sharing between services.
