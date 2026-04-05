"""Unit tests for event store backends."""

from src.event_store import create_event_store
from src.event_store.base import NullEventStore
from src.event_store.buffer import BufferedEventStore
from src.event_store.models import AgentPerformanceRecord, EventRecord, LLMCallRecord, ToolCallRecord
from src.event_store.postgres_store import PostgresEventStore
from src.event_store.sqlite_store import SQLiteEventStore


def test_create_event_store_disabled_returns_null_store() -> None:
    store = create_event_store({"enabled": False})
    assert isinstance(store, NullEventStore)


def test_sqlite_event_store_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store = create_event_store(
        {
            "enabled": True,
            "backend": "sqlite",
            "schema_version": 1,
            "sqlite": {"db_path": str(db_path)},
            "buffer": {"enabled": False},
        }
    )
    event = EventRecord(
        event_type="user_action",
        session_id="session-1",
        run_id="run-1",
        action="new_prompt",
        action_payload={"text": "growth"},
    )

    store.record(event)
    events = store.query(session_id="session-1", event_type="user_action", limit=10)

    assert len(events) == 1
    assert events[0].action == "new_prompt"
    assert events[0].action_payload == {"text": "growth"}
    store.close()


def test_create_event_store_unsupported_backend_raises() -> None:
    try:
        create_event_store({"enabled": True, "backend": "unknown"})
    except ValueError as exc:
        assert "Unsupported event store backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_create_event_store_postgres_requires_dsn() -> None:
    try:
        create_event_store({"enabled": True, "backend": "postgres", "postgres": {}})
    except ValueError as exc:
        assert "dsn or dsn_env_var" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_create_event_store_postgres_reads_dsn_env(monkeypatch) -> None:
    monkeypatch.setenv("EVENT_STORE_DSN", "postgresql://u:p@localhost:5432/db")

    store = create_event_store(
        {
            "enabled": True,
            "backend": "postgres",
            "postgres": {"dsn_env_var": "EVENT_STORE_DSN"},
            "buffer": {"enabled": False},
        }
    )

    assert isinstance(store, PostgresEventStore)
    store.close()


# ---------------------------------------------------------------------------
# NullEventStore — new MonitoringStore methods
# ---------------------------------------------------------------------------


def test_null_store_monitoring_methods_are_noops() -> None:
    store = NullEventStore()
    store.record_llm_call(
        LLMCallRecord(
            id="x",
            session_id="s",
            run_id=None,
            timestamp="2026-01-01T00:00:00.000+00:00",
            model=None,
            prompt=None,
            output=None,
            output_code=None,
            latency_ms=None,
            token_usage=None,
            temperature=None,
            max_tokens=None,
            stage=None,
            schema_version=1,
        )
    )
    store.record_tool_call(
        ToolCallRecord(
            id="x",
            session_id="s",
            run_id=None,
            timestamp="2026-01-01T00:00:00.000+00:00",
            tool_name="t",
            tool_call_id=None,
            arguments=None,
            result=None,
            agent_round=None,
            stage=None,
            schema_version=1,
        )
    )
    store.record_agent_performance(
        AgentPerformanceRecord(
            id="x",
            session_id="s",
            run_id="r",
            timestamp="2026-01-01T00:00:00.000+00:00",
            model=None,
            total_llm_calls=0,
            total_tool_calls=0,
            total_iterations=0,
            total_latency_ms=0.0,
            total_tokens=0,
            portfolio_return_1y=None,
            portfolio_current=None,
            portfolio_min=None,
            portfolio_max=None,
            tickers=[],
            weights={},
            status="completed",
            schema_version=1,
        )
    )
    assert store.query_llm_calls() == []
    assert store.query_tool_calls() == []
    assert store.query_agent_performance() == []


# ---------------------------------------------------------------------------
# BufferedEventStore — forwards MonitoringStore calls to backing store
# ---------------------------------------------------------------------------


def test_buffered_store_forwards_record_llm_call(tmp_path) -> None:
    backing = SQLiteEventStore(str(tmp_path / "events.db"))
    buffered = BufferedEventStore(backing, flush_interval_seconds=60, max_buffer_size=1000)

    rec = LLMCallRecord(
        id="buf-llm",
        session_id="s1",
        run_id="r1",
        timestamp="2026-01-01T00:00:00.000+00:00",
        model="test-model",
        prompt=None,
        output="hi",
        output_code=200,
        latency_ms=50.0,
        token_usage=None,
        temperature=0.2,
        max_tokens=100,
        stage="test",
        schema_version=1,
    )
    buffered.record_llm_call(rec)
    buffered.flush()  # records are buffered; flush before querying backing store
    results = buffered.query_llm_calls(session_id="s1")
    assert len(results) == 1
    assert results[0].id == "buf-llm"
    buffered.close()


def test_buffered_store_forwards_record_tool_call(tmp_path) -> None:
    backing = SQLiteEventStore(str(tmp_path / "events.db"))
    buffered = BufferedEventStore(backing, flush_interval_seconds=60, max_buffer_size=1000)

    rec = ToolCallRecord(
        id="buf-tool",
        session_id="s1",
        run_id="r1",
        timestamp="2026-01-01T00:00:00.000+00:00",
        tool_name="generate_tickers",
        tool_call_id=None,
        arguments={"n": 5},
        result={"tickers": ["AAPL"]},
        agent_round=1,
        stage="portfolio_agent",
        schema_version=1,
    )
    buffered.record_tool_call(rec)
    buffered.flush()  # records are buffered; flush before querying backing store
    results = buffered.query_tool_calls(session_id="s1")
    assert len(results) == 1
    assert results[0].tool_name == "generate_tickers"
    buffered.close()


def test_buffered_store_forwards_record_agent_performance(tmp_path) -> None:
    backing = SQLiteEventStore(str(tmp_path / "events.db"))
    buffered = BufferedEventStore(backing, flush_interval_seconds=60, max_buffer_size=1000)

    rec = AgentPerformanceRecord(
        id="buf-perf",
        session_id="s1",
        run_id="r1-unique",
        timestamp="2026-01-01T00:00:00.000+00:00",
        model="m",
        total_llm_calls=1,
        total_tool_calls=2,
        total_iterations=2,
        total_latency_ms=100.0,
        total_tokens=50,
        portfolio_return_1y=0.1,
        portfolio_current=1.1,
        portfolio_min=0.9,
        portfolio_max=1.2,
        tickers=["AAPL"],
        weights={"AAPL": 1.0},
        status="completed",
        schema_version=1,
    )
    buffered.record_agent_performance(rec)
    results = buffered.query_agent_performance(session_id="s1")
    assert len(results) == 1
    assert results[0].run_id == "r1-unique"
    buffered.close()


# ---------------------------------------------------------------------------
# SQLite WAL mode
# ---------------------------------------------------------------------------


def test_sqlite_event_store_uses_wal_journal_mode(tmp_path) -> None:
    """SQLiteEventStore must enable WAL mode for write performance."""
    store = SQLiteEventStore(str(tmp_path / "wal_test.db"))
    row = store._connection.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"
    store.close()


def test_sqlite_event_store_uses_normal_synchronous_mode(tmp_path) -> None:
    """SQLiteEventStore must set synchronous=NORMAL (1) for performance."""
    store = SQLiteEventStore(str(tmp_path / "sync_test.db"))
    row = store._connection.execute("PRAGMA synchronous").fetchone()
    # 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
    assert row[0] == 1
    store.close()
