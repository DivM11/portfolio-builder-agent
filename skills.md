# AI Coding Agent — Skills & Instructions

> **Role**: AI Engineer / Senior Software Developer  
> **Codebase**: Portfolio Builder Agent — a Streamlit + LLM tool-calling agent for US equity portfolio construction with structured monitoring.

---

## 1. Testability

### Testing Framework & Execution

- **Framework**: pytest with `pytest-cov` for coverage. Tests live in `tests/`, one file per source module (`test_<module>.py`).
- **Execution**: Always run tests via Docker:
  ```bash
  # Plain test run
  docker compose run --build --rm test

  # With coverage (enforces 90% minimum)
  docker compose run --build --rm test pytest --cov=src --cov-report=term-missing --cov-fail-under=90 -v --tb=short
  ```
  Never run pytest directly on the host.
- **Coverage threshold**: 90% minimum on `src/`. Configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`. PRs that drop coverage below 90% must not be merged.
- **Configuration**: Defined in `pyproject.toml` under `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `pythonpath = ["."]`.

### Test Design Rules

1. **One test file per source module.** If you create `src/foo.py`, create `tests/test_foo.py`.
2. **No test classes.** Use standalone `test_` functions. Group related tests with section comments:
   ```python
   # ---------------------------------------------------------------------------
   # Edge cases
   # ---------------------------------------------------------------------------
   ```
3. **No shared fixtures / no conftest.py.** Each test module is self-contained. Define helpers inline prefixed with `_`:
   ```python
   def _base_config():
       return {"agent": {"model": "test-model", "max_tokens": 100}}
   ```
4. **Arrange-Act-Assert (AAA).** Every test must have clear setup, execution, and assertion phases.
5. **Prefer custom mock classes over `unittest.mock`.** The codebase uses explicitly defined Dummy/Stub/Capture classes:
   - `DummyLLMService` — returns fixed responses.
   - `CaptureEventStore` — records calls in a list for later assertion.
   - `RecordingStore` — captures monitoring records.
   - Name mocks with intent: `Dummy*` (no-op), `Capture*` (records calls), `Stub*` (returns fixed data), `Error*` (raises).
6. **Use `monkeypatch`** (pytest built-in) for environment variables and module substitution — never mutate `os.environ` directly.
7. **Float comparisons** must use tolerance: `assert abs(actual - expected) < 0.01`.
8. **Test error paths explicitly.** Every public function that can fail should have a corresponding test for the failure mode (e.g., `test_json_parse_failure_fails_closed`).
9. **Every mock class must satisfy the Protocol it replaces.** If mocking `EventStore`, implement all Protocol methods — even as no-ops. This keeps `isinstance` checks valid at runtime.

### What to Test

| Layer | What to assert |
|-------|----------------|
| Tools (`src/tools/`) | Correct `work_state` mutations, valid return payloads, edge cases (empty input, missing tickers) |
| Agent (`agent.py`) | Tool-call sequencing, context preservation across rounds, guard blocking, refinement continuity |
| LLM Service | Event emission (LLMCallRecord recorded), error handling on API failure |
| Event Store | Write-then-read round-trips, query filters, upsert (ON CONFLICT) behavior |
| ETL | Aggregation arithmetic (total_latency, token sums), missing-field defaults |
| Config | Env var injection, missing key behavior, YAML defaults |
| Data transforms | Portfolio math (weight normalization, allocation rounding), summary compaction |

### What NOT to Test

- Streamlit rendering internals (use `DummyStreamlit` stubs for component wiring only).
- Third-party library internals (pandas, plotly, openai SDK).
- Private helper functions — test them through their public callers.

---

## 2. Style & Readability

### Formatting, Linting & Type Checking

All three tools are configured in `pyproject.toml`. Dev dependencies: `ruff`, `mypy`, `pytest-cov`.

#### Ruff (replaces black + flake8)

- **Linter**: `ruff check .` — runs pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, simplify, and print-statement detection.
- **Formatter**: `ruff format --check .` (dry-run) / `ruff format .` (apply).
- **Config** (`pyproject.toml`):
  - `target-version = "py311"`, `line-length = 120`.
  - Rule sets: `E, F, W, I, N, UP, B, SIM, T20`.
  - Quote style: double.
- Run before committing. Never add `# noqa` without a comment explaining why.

#### mypy (static type checking)

- **Command**: `mypy src/`
- **Config** (`pyproject.toml`):
  - `python_version = "3.11"`, `disallow_untyped_defs = true`, `warn_return_any = true`.
  - `ignore_missing_imports = true` (third-party stubs not required).
- Every new/modified function **must** pass mypy with `disallow_untyped_defs`. No `# type: ignore` without a justifying comment.

#### Running all checks locally (inside Docker)

```bash
# Lint
docker compose run --build --rm test ruff check .

# Format check (no changes)
docker compose run --build --rm test ruff format --check .

# Type check
docker compose run --build --rm test mypy src/

# Tests with coverage
docker compose run --build --rm test pytest --cov=src --cov-report=term-missing --cov-fail-under=90 -v --tb=short

# Plain tests
docker compose run --build --rm test
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | `PascalCase` | `PortfolioAgent`, `TickrDataManager` |
| Functions / methods | `snake_case` | `generate_tickers_tool`, `fetch_price_history_with_status` |
| Constants | `UPPER_SNAKE_CASE` | `HISTORY_STATUS_OK`, `TICKER_PATTERN` |
| Private / internal | Leading underscore | `_classify_history_exception`, `_run_loop` |
| Callbacks | Suffix `Callback` | `ProgressCallback`, `StepStatusCallback` |
| Manager classes | Suffix `Manager` | `TickrDataManager`, `TickrSummaryManager` |
| Record / model classes | Suffix `Record` or `Result` | `LLMCallRecord`, `AgentResult`, `InputGuardResult` |
| Test helpers | Leading underscore | `_base_config()`, `_make_llm()`, `_payload()` |

### Type Hints

- **Mandatory on all function signatures** — parameters and return types.
- Use Python 3.11+ syntax: `X | None` (not `Optional[X]`), `list[str]` (not `List[str]`).
- Use `dict[str, Any]` for loosely-typed dicts (config, JSON payloads).
- Use `Callable` with signatures for injected factories: `Callable[[str], RESTClient]`.

### Import Order

Import sorting is enforced by `ruff check` (rule `I` — isort-compatible).

1. `__future__` imports
2. Standard library (`logging`, `json`, `datetime`, `uuid`, `pathlib`, …)
3. Third-party (`pandas`, `openai`, `streamlit`, `omegaconf`, …)
4. Local (`src.*`)
5. Alphabetical within each group.

Auto-fix with `ruff check --fix .` if imports are out of order.

### Docstrings & Comments

- **Do not add docstrings** to functions/methods unless the logic is non-obvious or the function is part of a public Protocol.
- Use inline comments sparingly — only when the *why* isn't clear from the code.
- Section dividers in test files use the `# ---...---` pattern.

### Logging

- Use `logging` module, never `print()`.
- JSON-formatted logs (`logging.level`, `logging.format: "json"` in config).
- Attach `session_id` and `run_id` via `contextvars` — do not pass them manually through call chains.
- Log at appropriate levels: `logger.info` for lifecycle events, `logger.warning` for degraded paths, `logger.exception` inside `except` blocks.

---

## 3. Maintainability & Architecture

### Protocol-Based Interfaces

The codebase uses `typing.Protocol` with `@runtime_checkable` for all pluggable boundaries. Never use ABC/abstract classes.

#### EventStore Protocol (`src/event_store/base.py`)

```
EventStore (Protocol)
├── record(event: EventRecord) -> None
├── query(session_id, event_type, since, limit) -> list[EventRecord]
└── close() -> None
```

#### MonitoringStore Protocol (extends EventStore)

Adds specialized write/query methods for `LLMCallRecord`, `ToolCallRecord`, `AgentPerformanceRecord`.

#### Implementations

| Class | Purpose | Trade-off |
|-------|---------|-----------|
| `NullEventStore` | No-op, zero overhead | No observability |
| `SQLiteEventStore` | Full persistence, 4 tables | Single-writer concurrency limitation |
| `BufferedEventStore` | Wraps any EventStore, flushes on interval/threshold | Slight data loss risk on crash |
| `PostgresEventStore` | Placeholder for production scaling | Not yet implemented |

#### Design Rationale

- **Protocol over ABC**: Enables structural subtyping — any object with the right methods satisfies the contract, no inheritance required. Mocks in tests automatically comply.
- **`@runtime_checkable`**: Allows `isinstance()` guards at composition boundaries without import coupling.
- **Null Object pattern**: `NullEventStore` lets monitoring be completely disabled (`event_store.enabled: false`) with zero code changes.

### Dependency Injection

All major components accept dependencies through constructor parameters:

```python
PortfolioAgent(
    llm_service: LLMService,
    config: dict,
    event_store: EventStore | None = NullEventStore(),
    input_guard: InputGuard | None = None,
    tickr_data_manager: TickrDataManager | None = None,
    tickr_summary_manager: TickrSummaryManager | None = None,
    massive_client_factory: Callable = create_massive_client,
    stock_data_fetcher: Callable = fetch_stock_data,
)
```

**Rules:**
- Never instantiate collaborators inside a class. Accept them as constructor params.
- Use factory functions (`create_event_store`, `create_massive_client`) for complex construction — keep those in config or entry-point code.
- Default to Null Object or `None` for optional dependencies.
- Inject callables (factories, fetchers) instead of concrete instances when the dependency requires runtime parameters.

### Configuration

- **Single source of truth**: `config.yml` loaded via `OmegaConf`.
- **Secrets via environment variables only** — never in config files or source.
- Config is a plain `dict[str, Any]` once loaded (via `OmegaConf.to_container()`).
- Access nested values with standard dict access: `config["agent"]["model"]`.
- New config keys should be added to `config.yml` with sensible defaults. Document the key's purpose inline in YAML.

### Tool Pattern

Each tool in `src/tools/` follows a strict contract:

1. **`tool_definition() -> dict`** — returns an OpenAI-compatible function schema.
2. **`<tool_name>_tool(arguments, config, ...dependencies) -> str`** — executes the tool, updates `work_state`, returns a string result for the LLM message history.
3. Tool dispatch lives in `agent.py:_execute_tool()` as a name-matched switch.

**When adding a new tool:**
1. Create `src/tools/<tool_name>.py` with `tool_definition()` and `<tool_name>_tool()`.
2. Register the tool definition in the agent's tool list.
3. Add a dispatch branch in `_execute_tool()`.
4. Add a corresponding `tests/test_<tool_name>.py` (or extend `test_agent.py`).

### Context Management

Three distinct context objects — respect their boundaries:

| Class | Mutability | Purpose |
|-------|-----------|---------|
| `AgentContext` | Frozen dataclass | Immutable snapshot for logging/auditing. Never mutate. |
| `Context` | Mutable | Carries `messages`, `work_state`, `tool_invocations` across rounds. Owned by the agent run. |
| `AgentResult` | Mutable dataclass | Final output. Backfilled from `work_state` if LLM response is incomplete. |

**Rules:**
- Never store mutable state in `AgentContext` — it's a snapshot.
- `work_state` is the shared scratchpad between tools. Tools read from and write to it.
- `Context.prepare_for_refine()` seeds state for multi-turn refinement — always go through it, don't manually manipulate messages.

### Coupling Boundaries

**Intentionally loose:**
- Event recording is fully optional (Null Object pattern).
- Input guard is optional (`None` disables it).
- Monitoring API is a separate FastAPI process reading the same SQLite — no runtime coupling with the agent.
- UI callbacks (`ProgressCallback`, `StepStatusCallback`) are optional callables — agent works without them.

**Intentionally tight (and acceptable):**
- Five tools are hardcoded in the dispatch switch — this is a deliberate constraint, not an oversight.
- `AgentResult` JSON schema is a contract between the LLM system prompt and the parsing logic — changing one requires changing the other.
- Config YAML structure is tightly coupled to `dashboard.py` UI text — this is the "single source of truth" trade-off.

**When deciding whether to abstract:**
- Abstract only when there are **two or more concrete implementations** (or a clear, imminent need for a second).
- Don't create interfaces for things that will never be swapped (e.g., the Streamlit dashboard).
- Prefer injecting callables over building interface hierarchies for single-method contracts.

---

## 4. General Agent Instructions

### Before Writing Code

1. **Read the relevant source files first.** Understand existing patterns before proposing changes. Never generate code that contradicts the established architecture.
2. **Check `config.yml`** for any configurable value before hardcoding. If the value should be configurable, add it to config.
3. **Check the Protocol** that governs the module you're modifying. Ensure your changes satisfy the existing contract — or explicitly extend the Protocol if needed.
4. **Trace the data flow.** Understand how data moves from user input → agent → tools → event store → monitoring API before touching any layer.

### Writing Code

5. **No over-engineering.** Don't add abstractions, helpers, or defensive code for cases that don't exist yet. Solve the problem at hand.
6. **No custom exception classes** unless the error needs to cross a system boundary with structured information. Use `ValueError` / `TypeError` with clear messages.
7. **Use `Decimal`** for financial calculations involving money (portfolio allocation). Use `float` for statistical measures (returns, ratios).
8. **JSON compactness matters.** When serializing data for LLM context windows, use `separators=(',', ':')` to minimize token usage.
9. **Callbacks for progress reporting.** Never call `st.write()` or `print()` from agent/tool code. Accept optional callbacks and invoke them if provided.
10. **Thread safety in event stores.** `BufferedEventStore` uses `collections.deque` and `threading.Timer`. If adding buffering or async behavior, use thread-safe primitives.
11. **Upsert pattern for ETL.** `agent_performance` uses `ON CONFLICT(run_id) DO UPDATE`. Follow this pattern for any materialised/aggregated table.

### Error Handling

12. **Fail closed for security-sensitive paths.** If `InputGuard` cannot parse the LLM classification, treat the input as unsafe.
13. **Broad `except Exception` with `logger.exception()`** at service boundaries (API handlers, agent run entry points). Let the exception propagate in internal code.
14. **Classify errors, don't swallow them.** See `_classify_history_exception()` — return a status string that callers can act on, instead of silently returning `None`.
15. **Graceful degradation for missing data.** If a ticker fetch fails, skip it and continue with the rest. Log a warning. Never crash the entire run.

### Data Models & Records

16. **All event records** must include: `id` (UUID), `session_id`, `run_id`, `timestamp` (ISO with milliseconds), `schema_version`.
17. **Schema version** every record model. Increment `schema_version` when adding fields. Never remove fields — add new ones alongside.
18. **Timestamp format**: ISO 8601 with millisecond precision. Use `datetime.utcnow().isoformat(timespec="milliseconds")`.

### Security & Input Validation

19. **Validate at system boundaries only.** Ticker symbols are validated with regex `^[A-Z][A-Z0-9.-]{0,9}$` at ingestion. Internal code trusts sanitized data.
20. **Never log raw API keys or secrets.** Config loading masks secret values. If adding new secrets, follow the same `key_env_var` indirection pattern.
21. **LLM output is untrusted.** Always parse and validate JSON returned by the LLM. Use `llm_validation.py` extraction functions — don't trust raw text.
22. **SQL parameterization.** All SQLite queries use parameterized queries (`?` placeholders). Never interpolate user input into SQL strings.

### Performance Considerations

23. **Cache aggressively in managers.** `TickrDataManager` caches per-ticker payloads and `TickrSummaryManager` caches by `(ticker_set, cache_version)`. Only refetch/rebuild when inputs change.
24. **Minimize LLM token usage.** Compact summaries (`{"t":"AAPL","p":{"min":"$100",...}}`), limit message history, use `max_tokens` from config.
25. **Buffer event writes.** High-frequency events (LLM calls, tool calls) should go through `BufferedEventStore`. Don't write synchronously in the hot path.

### Docker & Deployment

26. **All tests and checks run through Docker.** Never assume the host has Python or dependencies installed.
    ```bash
    docker compose run --build --rm test                # plain pytest
    docker compose run --build --rm test ruff check .   # lint
    docker compose run --build --rm test ruff format --check .  # format check
    docker compose run --build --rm test mypy src/      # type check
    docker compose run --build --rm test pytest --cov=src --cov-fail-under=90 -v --tb=short  # coverage
    ```
27. **Three-service architecture**: `app` (8501), `monitor-api` (8000), `monitor-ui` (8502). They share a Docker volume for the SQLite database. Changes to the event schema affect all three.
28. **Environment variables** are supplied via `.secrets` file (git-ignored) and read by Docker Compose. Never commit secrets.

### Code Review Checklist

Before considering any change complete, verify:

- [ ] `ruff check .` passes (no lint errors).
- [ ] `ruff format --check .` passes (code is formatted).
- [ ] `mypy src/` passes (no type errors).
- [ ] `docker compose run --build --rm test` passes (all tests green).
- [ ] `pytest --cov=src --cov-fail-under=90` passes (coverage ≥ 90%).
- [ ] New/modified functions have type hints on all parameters and return types.
- [ ] New source modules have corresponding test files.
- [ ] Tests use custom mock classes (not `unittest.mock.patch`) for major dependencies.
- [ ] No hardcoded values that belong in `config.yml`.
- [ ] Protocol contracts are still satisfied after the change.
- [ ] Event records include all required fields (`id`, `session_id`, `run_id`, `timestamp`, `schema_version`).
- [ ] No secrets, API keys, or PII in source code or logs.
