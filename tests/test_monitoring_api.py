"""Unit tests for monitoring API query helpers and table endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException

import src.monitoring_api as monitoring_api

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_monitoring_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id TEXT,
                session_id TEXT,
                run_id TEXT,
                event_type TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE llm_calls (
                id TEXT,
                session_id TEXT,
                run_id TEXT,
                stage TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tool_calls (
                id TEXT,
                session_id TEXT,
                run_id TEXT,
                tool_name TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE agent_performance (
                id TEXT,
                session_id TEXT,
                run_id TEXT,
                status TEXT,
                timestamp TEXT
            )
            """
        )

        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            ("e1", "s1", "r1", "llm_request", "2026-01-01T00:00:00.000+00:00"),
        )
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            ("e2", "s2", "r2", "llm_response", "2026-01-01T00:00:01.000+00:00"),
        )

        conn.execute(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?)",
            ("l1", "s1", "r1", "portfolio_agent", "2026-01-01T00:00:02.000+00:00"),
        )
        conn.execute(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?)",
            ("l2", "s1", "r2", "input_guard", "2026-01-01T00:00:03.000+00:00"),
        )

        conn.execute(
            "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?)",
            ("t1", "s1", "r1", "generate_tickers", "2026-01-01T00:00:04.000+00:00"),
        )
        conn.execute(
            "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?)",
            ("t2", "s2", "r2", "fetch_ticker_data", "2026-01-01T00:00:05.000+00:00"),
        )

        conn.execute(
            "INSERT INTO agent_performance VALUES (?, ?, ?, ?, ?)",
            ("p1", "s1", "r1", "completed", "2026-01-01T00:00:06.000+00:00"),
        )
        conn.execute(
            "INSERT INTO agent_performance VALUES (?, ?, ?, ?, ?)",
            ("p2", "s2", "r2", "error", "2026-01-01T00:00:07.000+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_where_builds_expected_clause() -> None:
    assert monitoring_api._where([]) == ""
    assert monitoring_api._where(["session_id = ?", "event_type = ?"]) == "WHERE session_id = ? AND event_type = ?"


def test_connect_raises_when_db_missing(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.db"
    monkeypatch.setattr(monitoring_api, "_DB_PATH", missing)

    with pytest.raises(HTTPException) as exc_info:
        monitoring_api._connect()

    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_health_reports_database_presence(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    health_before = monitoring_api.health()
    assert health_before["status"] == "ok"
    assert health_before["db_exists"] is False

    _create_monitoring_db(db_path)
    health_after = monitoring_api.health()
    assert health_after["db_exists"] is True


def test_get_events_filters_by_session_and_type(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_events(session_id="s1", event_type="llm_request", limit=10)

    assert len(rows) == 1
    assert rows[0]["id"] == "e1"


def test_get_llm_calls_filters_stage_and_limit(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_llm_calls(session_id="s1", run_id=None, stage="portfolio_agent", limit=1)

    assert len(rows) == 1
    assert rows[0]["id"] == "l1"


def test_get_tool_calls_filters_tool_name(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_tool_calls(session_id=None, run_id=None, tool_name="fetch_ticker_data", limit=10)

    assert len(rows) == 1
    assert rows[0]["id"] == "t2"


def test_get_tool_calls_filters_by_run_id(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_tool_calls(session_id=None, run_id="r1", tool_name=None, limit=10)

    assert len(rows) == 1
    assert rows[0]["id"] == "t1"


def test_get_agent_performance_filters_status(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_agent_performance(session_id=None, run_id=None, status="completed", limit=10)

    assert len(rows) == 1
    assert rows[0]["id"] == "p1"


def test_get_agent_performance_filters_session_and_run_id(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "events.db"
    _create_monitoring_db(db_path)
    monkeypatch.setattr(monitoring_api, "_DB_PATH", db_path)

    rows = monitoring_api.get_agent_performance(session_id="s2", run_id="r2", status=None, limit=10)

    assert len(rows) == 1
    assert rows[0]["id"] == "p2"
