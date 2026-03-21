"""Tests for MonitoringStore methods on SQLiteEventStore and NullEventStore."""

from __future__ import annotations

import pytest

from src.event_store.base import NullEventStore
from src.event_store.models import AgentPerformanceRecord, LLMCallRecord, ToolCallRecord
from src.event_store.sqlite_store import SQLiteEventStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_record(session_id: str = "s1", run_id: str = "r1", **kwargs) -> LLMCallRecord:
    return LLMCallRecord(
        id=kwargs.pop("id", "llm-id-1"),
        session_id=session_id,
        run_id=run_id,
        timestamp="2026-01-01T00:00:00.000+00:00",
        model="anthropic/claude-3.5-haiku",
        prompt=[{"role": "user", "content": "hi"}],
        output="hello",
        output_code=200,
        latency_ms=123.4,
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        temperature=0.2,
        max_tokens=512,
        stage="portfolio_agent",
        schema_version=1,
        **kwargs,
    )


def _tool_record(session_id: str = "s1", run_id: str = "r1", **kwargs) -> ToolCallRecord:
    return ToolCallRecord(
        id=kwargs.pop("id", "tool-id-1"),
        session_id=session_id,
        run_id=run_id,
        timestamp="2026-01-01T00:00:01.000+00:00",
        tool_name="generate_tickers",
        tool_call_id="tc-1",
        arguments={"count": 5},
        result={"valid_tickers": ["AAPL"]},
        agent_round=1,
        stage="portfolio_agent",
        schema_version=1,
        **kwargs,
    )


def _perf_record(session_id: str = "s1", run_id: str = "r1", **kwargs) -> AgentPerformanceRecord:
    return AgentPerformanceRecord(
        id=kwargs.pop("id", "perf-id-1"),
        session_id=session_id,
        run_id=kwargs.pop("run_id", run_id),
        timestamp=kwargs.pop("timestamp", "2026-01-01T00:00:02.000+00:00"),
        model=kwargs.pop("model", "anthropic/claude-3.5-haiku"),
        total_llm_calls=kwargs.pop("total_llm_calls", 2),
        total_tool_calls=kwargs.pop("total_tool_calls", 5),
        total_iterations=kwargs.pop("total_iterations", 5),
        total_latency_ms=kwargs.pop("total_latency_ms", 456.7),
        total_tokens=kwargs.pop("total_tokens", 100),
        portfolio_return_1y=kwargs.pop("portfolio_return_1y", 0.15),
        portfolio_current=kwargs.pop("portfolio_current", 1.15),
        portfolio_min=kwargs.pop("portfolio_min", 0.95),
        portfolio_max=kwargs.pop("portfolio_max", 1.20),
        tickers=kwargs.pop("tickers", ["AAPL", "MSFT"]),
        weights=kwargs.pop("weights", {"AAPL": 0.6, "MSFT": 0.4}),
        status=kwargs.pop("status", "completed"),
        error_message=kwargs.pop("error_message", None),
        schema_version=kwargs.pop("schema_version", 1),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SQLiteEventStore — table creation
# ---------------------------------------------------------------------------

def test_sqlite_creates_llm_calls_table(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    tables = {
        row[0]
        for row in store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "llm_calls" in tables
    store.close()


def test_sqlite_creates_tool_calls_table(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    tables = {
        row[0]
        for row in store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "tool_calls" in tables
    store.close()


def test_sqlite_creates_agent_performance_table(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    tables = {
        row[0]
        for row in store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "agent_performance" in tables
    store.close()


# ---------------------------------------------------------------------------
# SQLiteEventStore — llm_calls round-trip
# ---------------------------------------------------------------------------

def test_record_and_query_llm_call(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    rec = _llm_record()
    store.record_llm_call(rec)

    results = store.query_llm_calls(session_id="s1", run_id="r1")
    assert len(results) == 1
    r = results[0]
    assert r.id == "llm-id-1"
    assert r.model == "anthropic/claude-3.5-haiku"
    assert r.prompt == [{"role": "user", "content": "hi"}]
    assert r.output == "hello"
    assert r.output_code == 200
    assert abs(r.latency_ms - 123.4) < 0.01
    assert r.token_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert r.stage == "portfolio_agent"
    store.close()


def test_query_llm_calls_filters_by_run_id(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    store.record_llm_call(_llm_record(session_id="s1", run_id="r1", id="id1"))
    store.record_llm_call(_llm_record(session_id="s1", run_id="r2", id="id2"))

    results = store.query_llm_calls(session_id="s1", run_id="r1")
    assert len(results) == 1
    assert results[0].run_id == "r1"
    store.close()


def test_query_llm_calls_no_filters_returns_all(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    store.record_llm_call(_llm_record(session_id="s1", run_id="r1", id="id1"))
    store.record_llm_call(_llm_record(session_id="s2", run_id="r2", id="id2"))

    results = store.query_llm_calls(limit=10)
    assert len(results) == 2
    store.close()


def test_llm_call_null_fields_stored_correctly(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    rec = LLMCallRecord(
        id="llm-null",
        session_id="s1",
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
    store.record_llm_call(rec)
    results = store.query_llm_calls(session_id="s1")
    assert len(results) == 1
    assert results[0].model is None
    assert results[0].prompt is None
    store.close()


# ---------------------------------------------------------------------------
# SQLiteEventStore — tool_calls round-trip
# ---------------------------------------------------------------------------

def test_record_and_query_tool_call(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    rec = _tool_record()
    store.record_tool_call(rec)

    results = store.query_tool_calls(session_id="s1", run_id="r1")
    assert len(results) == 1
    r = results[0]
    assert r.id == "tool-id-1"
    assert r.tool_name == "generate_tickers"
    assert r.arguments == {"count": 5}
    assert r.result == {"valid_tickers": ["AAPL"]}
    assert r.agent_round == 1
    assert r.stage == "portfolio_agent"
    store.close()


def test_query_tool_calls_filters_by_session(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    store.record_tool_call(_tool_record(session_id="s1", id="t1"))
    store.record_tool_call(_tool_record(session_id="s2", id="t2"))

    results = store.query_tool_calls(session_id="s1")
    assert len(results) == 1
    assert results[0].session_id == "s1"
    store.close()


# ---------------------------------------------------------------------------
# SQLiteEventStore — agent_performance round-trip
# ---------------------------------------------------------------------------

def test_record_and_query_agent_performance(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    rec = _perf_record()
    store.record_agent_performance(rec)

    results = store.query_agent_performance(session_id="s1", run_id="r1")
    assert len(results) == 1
    r = results[0]
    assert r.run_id == "r1"
    assert r.total_llm_calls == 2
    assert r.total_tool_calls == 5
    assert r.total_iterations == 5
    assert r.total_tokens == 100
    assert abs(r.portfolio_return_1y - 0.15) < 0.001
    assert r.tickers == ["AAPL", "MSFT"]
    assert r.weights == {"AAPL": 0.6, "MSFT": 0.4}
    assert r.status == "completed"
    store.close()


def test_agent_performance_upserts_on_duplicate_run_id(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    rec1 = _perf_record(status="completed")
    store.record_agent_performance(rec1)

    rec2 = _perf_record(id="perf-id-2", status="error", error_message="boom")
    store.record_agent_performance(rec2)

    results = store.query_agent_performance(run_id="r1")
    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error_message == "boom"
    store.close()


def test_query_agent_performance_filters_by_run_id(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "events.db"))
    store.record_agent_performance(_perf_record(run_id="r1", id="p1"))
    store.record_agent_performance(_perf_record(run_id="r2", id="p2"))

    results = store.query_agent_performance(run_id="r1")
    assert len(results) == 1
    assert results[0].run_id == "r1"
    store.close()


# ---------------------------------------------------------------------------
# NullEventStore — new methods are no-ops
# ---------------------------------------------------------------------------

def test_null_event_store_record_llm_call_is_noop() -> None:
    store = NullEventStore()
    store.record_llm_call(_llm_record())  # should not raise
    assert store.query_llm_calls() == []


def test_null_event_store_record_tool_call_is_noop() -> None:
    store = NullEventStore()
    store.record_tool_call(_tool_record())
    assert store.query_tool_calls() == []


def test_null_event_store_record_agent_performance_is_noop() -> None:
    store = NullEventStore()
    store.record_agent_performance(_perf_record())
    assert store.query_agent_performance() == []


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------

def test_llm_call_record_to_dict() -> None:
    rec = _llm_record()
    d = rec.to_dict()
    assert d["id"] == "llm-id-1"
    assert d["stage"] == "portfolio_agent"
    assert isinstance(d["prompt"], list)


def test_tool_call_record_to_dict() -> None:
    rec = _tool_record()
    d = rec.to_dict()
    assert d["tool_name"] == "generate_tickers"
    assert d["arguments"] == {"count": 5}


def test_agent_performance_record_to_dict() -> None:
    rec = _perf_record()
    d = rec.to_dict()
    assert d["status"] == "completed"
    assert d["tickers"] == ["AAPL", "MSFT"]
