"""Tests for the ETL materialise_agent_performance function."""

from __future__ import annotations

from src.agent_models import AgentResult
from src.etl.agent_performance import materialise_agent_performance
from src.event_store.base import NullEventStore
from src.event_store.models import AgentPerformanceRecord, LLMCallRecord, ToolCallRecord


class CaptureMixedStore(NullEventStore):
    """In-memory store that captures all monitoring record types."""

    def __init__(self) -> None:
        self._llm: list[LLMCallRecord] = []
        self._tools: list[ToolCallRecord] = []
        self._perfs: list[AgentPerformanceRecord] = []

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._llm.append(record)

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._tools.append(record)

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        self._perfs.append(record)

    def query_llm_calls(self, *, session_id=None, run_id=None, limit=100) -> list[LLMCallRecord]:
        results = self._llm
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if run_id:
            results = [r for r in results if r.run_id == run_id]
        return results[:limit]

    def query_tool_calls(self, *, session_id=None, run_id=None, limit=100) -> list[ToolCallRecord]:
        results = self._tools
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if run_id:
            results = [r for r in results if r.run_id == run_id]
        return results[:limit]

    def query_agent_performance(self, *, session_id=None, run_id=None, limit=100) -> list[AgentPerformanceRecord]:
        return self._perfs[:limit]


def _make_llm(session_id: str, run_id: str, latency_ms: float, total_tokens: int) -> LLMCallRecord:
    return LLMCallRecord(
        id="l1",
        session_id=session_id,
        run_id=run_id,
        timestamp="2026-01-01T00:00:00.000+00:00",
        model="anthropic/test",
        prompt=None,
        output="ok",
        output_code=200,
        latency_ms=latency_ms,
        token_usage={"total_tokens": total_tokens},
        temperature=0.2,
        max_tokens=512,
        stage="portfolio_agent",
        schema_version=1,
    )


def _make_tool(session_id: str, run_id: str, agent_round: int) -> ToolCallRecord:
    return ToolCallRecord(
        id="t1",
        session_id=session_id,
        run_id=run_id,
        timestamp="2026-01-01T00:00:01.000+00:00",
        tool_name="generate_tickers",
        tool_call_id="tc1",
        arguments={},
        result={},
        agent_round=agent_round,
        stage="portfolio_agent",
        schema_version=1,
    )


def _result(tickers=None, weights=None) -> AgentResult:
    r = AgentResult()
    r.tickers = tickers or ["AAPL", "MSFT"]
    r.weights = weights or {"AAPL": 0.6, "MSFT": 0.4}
    return r


# ---------------------------------------------------------------------------
# Core aggregation logic
# ---------------------------------------------------------------------------


def test_materialise_aggregates_llm_calls() -> None:
    store = CaptureMixedStore()
    store.record_llm_call(_make_llm("s1", "r1", latency_ms=100.0, total_tokens=50))
    store.record_llm_call(_make_llm("s1", "r1", latency_ms=200.0, total_tokens=80))

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={"return_1y": 0.1, "current": 1.1, "min": 0.9, "max": 1.2},
        model="anthropic/test",
    )

    assert rec.total_llm_calls == 2
    assert abs(rec.total_latency_ms - 300.0) < 0.01
    assert rec.total_tokens == 130


def test_materialise_aggregates_tool_calls_and_max_round() -> None:
    store = CaptureMixedStore()
    store.record_tool_call(_make_tool("s1", "r1", agent_round=1))
    store.record_tool_call(_make_tool("s1", "r1", agent_round=2))
    store.record_tool_call(_make_tool("s1", "r1", agent_round=3))

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="anthropic/test",
    )

    assert rec.total_tool_calls == 3
    assert rec.total_iterations == 3


def test_materialise_merges_portfolio_stats() -> None:
    store = CaptureMixedStore()
    stats = {"return_1y": 0.25, "current": 1.25, "min": 0.85, "max": 1.30}

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats=stats,
        model="m",
    )

    assert abs(rec.portfolio_return_1y - 0.25) < 0.001
    assert abs(rec.portfolio_current - 1.25) < 0.001
    assert abs(rec.portfolio_min - 0.85) < 0.001
    assert abs(rec.portfolio_max - 1.30) < 0.001


def test_materialise_calls_record_agent_performance() -> None:
    store = CaptureMixedStore()

    materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
    )

    assert len(store._perfs) == 1
    assert store._perfs[0].run_id == "r1"
    assert store._perfs[0].session_id == "s1"


def test_materialise_returns_record_with_correct_fields() -> None:
    store = CaptureMixedStore()
    result = _result(tickers=["AAPL"], weights={"AAPL": 1.0})

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=result,
        portfolio_stats={},
        model="anthropic/test",
    )

    assert rec.tickers == ["AAPL"]
    assert rec.weights == {"AAPL": 1.0}
    assert rec.model == "anthropic/test"
    assert rec.status == "completed"


def test_materialise_status_error() -> None:
    store = CaptureMixedStore()

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
        status="error",
        error_message="timeout",
    )

    assert rec.status == "error"
    assert rec.error_message == "timeout"


def test_materialise_status_guard_blocked() -> None:
    store = CaptureMixedStore()

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
        status="guard_blocked",
    )

    assert rec.status == "guard_blocked"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_materialise_zero_llm_calls() -> None:
    store = CaptureMixedStore()

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
    )

    assert rec.total_llm_calls == 0
    assert rec.total_latency_ms == 0.0
    assert rec.total_tokens == 0


def test_materialise_zero_tool_calls() -> None:
    store = CaptureMixedStore()

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
    )

    assert rec.total_tool_calls == 0
    assert rec.total_iterations == 0


def test_materialise_missing_portfolio_stats() -> None:
    store = CaptureMixedStore()

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
    )

    assert rec.portfolio_return_1y is None
    assert rec.portfolio_current is None
    assert rec.portfolio_min is None
    assert rec.portfolio_max is None


def test_materialise_llm_call_with_none_latency() -> None:
    """LLM calls with no latency should be treated as 0 in aggregation."""
    store = CaptureMixedStore()
    llm = _make_llm("s1", "r1", latency_ms=100.0, total_tokens=10)
    llm_no_latency = LLMCallRecord(
        id="l2",
        session_id="s1",
        run_id="r1",
        timestamp="2026-01-01T00:00:00.000+00:00",
        model="m",
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
    store.record_llm_call(llm)
    store.record_llm_call(llm_no_latency)

    rec = materialise_agent_performance(
        store,
        session_id="s1",
        run_id="r1",
        result=_result(),
        portfolio_stats={},
        model="m",
    )

    assert abs(rec.total_latency_ms - 100.0) < 0.01
    assert rec.total_tokens == 10
