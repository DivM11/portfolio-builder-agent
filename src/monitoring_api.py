"""FastAPI monitoring API for the Portfolio Builder Agent.

Provides read-only HTTP endpoints to query the four SQLite monitoring tables:
  - events            (legacy EventRecord log)
  - llm_calls         (one row per LLM HTTP round-trip)
  - tool_calls        (one row per agent tool invocation)
  - agent_performance (ETL-materialised summary per run)

Run via docker-compose (monitor-api service) or directly:
    uvicorn src.monitoring_api:app --host 0.0.0.0 --port 8000

Environment variables:
    DB_PATH   Path to the SQLite database  (default: data/events.db)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Portfolio Agent Monitoring API",
    version="1.0.0",
    description=("Read-only query interface for the SQLite event store. Interactive docs available at /docs."),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_DB_PATH = Path(os.environ.get("DB_PATH", "data/events.db"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    if not _DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at '{_DB_PATH}'. Run the agent first to create it.",
        )
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _where(clauses: list[str]) -> str:
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness check — returns DB path and whether the file exists."""
    return {
        "status": "ok",
        "db_path": str(_DB_PATH),
        "db_exists": _DB_PATH.exists(),
    }


# ---------------------------------------------------------------------------
# Table endpoints
# ---------------------------------------------------------------------------


@app.get("/events", tags=["tables"])
def get_events(
    session_id: str | None = Query(None, description="Filter by session_id"),
    event_type: str | None = Query(None, description="Filter by event_type"),
    limit: int = Query(200, ge=1, le=5000, description="Maximum rows to return"),
) -> list[dict[str, Any]]:
    """Return rows from the legacy **events** table."""
    clauses: list[str] = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    params.append(limit)
    return _fetch(
        f"SELECT * FROM events {_where(clauses)} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )


@app.get("/llm-calls", tags=["tables"])
def get_llm_calls(
    session_id: str | None = Query(None),
    run_id: str | None = Query(None),
    stage: str | None = Query(None, description="Filter by stage (e.g. portfolio_agent)"),
    limit: int = Query(200, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Return rows from the **llm_calls** table (one row per LLM HTTP call)."""
    clauses: list[str] = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    params.append(limit)
    return _fetch(
        f"SELECT * FROM llm_calls {_where(clauses)} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )


@app.get("/tool-calls", tags=["tables"])
def get_tool_calls(
    session_id: str | None = Query(None),
    run_id: str | None = Query(None),
    tool_name: str | None = Query(None, description="Filter by tool name (e.g. generate_tickers)"),
    limit: int = Query(200, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Return rows from the **tool_calls** table (one row per agent tool invocation)."""
    clauses: list[str] = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if tool_name:
        clauses.append("tool_name = ?")
        params.append(tool_name)
    params.append(limit)
    return _fetch(
        f"SELECT * FROM tool_calls {_where(clauses)} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )


@app.get("/agent-performance", tags=["tables"])
def get_agent_performance(
    session_id: str | None = Query(None),
    run_id: str | None = Query(None),
    status: str | None = Query(None, description="completed | error | guard_blocked"),
    limit: int = Query(200, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Return rows from the **agent_performance** table (one ETL row per completed run)."""
    clauses: list[str] = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    params.append(limit)
    return _fetch(
        f"SELECT * FROM agent_performance {_where(clauses)} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )
