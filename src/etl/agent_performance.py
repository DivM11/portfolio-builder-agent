"""ETL: aggregate llm_calls + tool_calls into agent_performance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.agent_models import AgentResult
from src.event_store.base import MonitoringStore
from src.event_store.models import AgentPerformanceRecord

logger = logging.getLogger(__name__)


def materialise_agent_performance(
    store: MonitoringStore,
    *,
    session_id: str,
    run_id: str,
    result: AgentResult,
    portfolio_stats: dict[str, Any],
    model: str,
    schema_version: int = 1,
    status: str = "completed",
    error_message: str | None = None,
) -> AgentPerformanceRecord:
    """Query llm_calls and tool_calls, aggregate, then upsert agent_performance.

    Args:
        store: A MonitoringStore that supports the monitoring query/record methods.
        session_id: The user session identifier.
        run_id: The agent invocation identifier (unique per run).
        result: The AgentResult returned by the agent.
        portfolio_stats: Output of summarize_portfolio_stats() or empty dict.
        model: The primary model name used during the run.
        schema_version: Schema version to tag the record with.
        status: One of 'completed', 'error', 'guard_blocked'.
        error_message: Set when status != 'completed'.

    Returns:
        The persisted AgentPerformanceRecord.
    """
    llm_calls = store.query_llm_calls(session_id=session_id, run_id=run_id, limit=10_000)
    tool_calls = store.query_tool_calls(session_id=session_id, run_id=run_id, limit=10_000)

    total_llm_calls = len(llm_calls)
    total_tool_calls = len(tool_calls)
    total_latency_ms = sum(c.latency_ms or 0.0 for c in llm_calls)
    total_tokens = sum(
        int((c.token_usage or {}).get("total_tokens", 0))
        for c in llm_calls
    )
    total_iterations = max((c.agent_round or 0 for c in tool_calls), default=0)

    record = AgentPerformanceRecord(
        id=str(uuid4()),
        session_id=session_id,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        model=model or None,
        total_llm_calls=total_llm_calls,
        total_tool_calls=total_tool_calls,
        total_iterations=total_iterations,
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        portfolio_return_1y=portfolio_stats.get("return_1y"),
        portfolio_current=portfolio_stats.get("current"),
        portfolio_min=portfolio_stats.get("min"),
        portfolio_max=portfolio_stats.get("max"),
        tickers=list(result.tickers),
        weights=dict(result.weights),
        status=status,
        error_message=error_message,
        schema_version=schema_version,
    )

    store.record_agent_performance(record)
    logger.info(
        "[session=%s run=%s] agent_performance materialised: status=%s llm_calls=%d tool_calls=%d iterations=%d",
        session_id,
        run_id,
        status,
        total_llm_calls,
        total_tool_calls,
        total_iterations,
    )
    return record
