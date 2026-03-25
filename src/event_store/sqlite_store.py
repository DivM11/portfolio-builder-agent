"""SQLite-backed event store implementation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.event_store.models import AgentPerformanceRecord, EventRecord, LLMCallRecord, ToolCallRecord


class SQLiteEventStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent write performance
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                run_id TEXT,
                event_type TEXT NOT NULL,
                request_name TEXT,
                model TEXT,
                temperature REAL,
                max_tokens INTEGER,
                messages TEXT,
                raw_output TEXT,
                status_code INTEGER,
                latency_ms REAL,
                token_usage TEXT,
                parsed_output TEXT,
                validation_errors TEXT,
                action TEXT,
                action_payload TEXT,
                tool_name TEXT,
                tool_arguments TEXT,
                tool_result TEXT,
                tool_call_id TEXT,
                agent TEXT,
                iteration INTEGER,
                agent_round INTEGER
            )
            """
        )
        self._ensure_columns(
            {
                "tool_name": "TEXT",
                "tool_arguments": "TEXT",
                "tool_result": "TEXT",
                "tool_call_id": "TEXT",
                "agent_round": "INTEGER",
            }
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")
        self._connection.commit()

        self._initialize_llm_calls()
        self._initialize_tool_calls()
        self._initialize_agent_performance()

    def _initialize_llm_calls(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT,
                timestamp TEXT NOT NULL,
                model TEXT,
                prompt TEXT,
                output TEXT,
                output_code INTEGER,
                latency_ms REAL,
                token_usage TEXT,
                temperature REAL,
                max_tokens INTEGER,
                stage TEXT,
                schema_version INTEGER NOT NULL
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(timestamp)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_stage ON llm_calls(stage)")
        self._connection.commit()

    def _initialize_tool_calls(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT,
                timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_call_id TEXT,
                arguments TEXT,
                result TEXT,
                agent_round INTEGER,
                stage TEXT,
                schema_version INTEGER NOT NULL
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(timestamp)")
        self._connection.commit()

    def _initialize_agent_performance(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                model TEXT,
                total_llm_calls INTEGER NOT NULL,
                total_tool_calls INTEGER NOT NULL,
                total_iterations INTEGER NOT NULL,
                total_latency_ms REAL NOT NULL,
                total_tokens INTEGER NOT NULL,
                portfolio_return_1y REAL,
                portfolio_current REAL,
                portfolio_min REAL,
                portfolio_max REAL,
                tickers TEXT,
                weights TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                schema_version INTEGER NOT NULL
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf_session ON agent_performance(session_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf_ts ON agent_performance(timestamp)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf_status ON agent_performance(status)")
        self._connection.commit()

    def _ensure_columns(self, required: dict[str, str]) -> None:
        rows = self._connection.execute("PRAGMA table_info(events)").fetchall()
        existing = {str(row[1]) for row in rows}
        for name, col_type in required.items():
            if name not in existing:
                self._connection.execute(f"ALTER TABLE events ADD COLUMN {name} {col_type}")

    @staticmethod
    def _dumps(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value)

    @staticmethod
    def _loads(value: str | None) -> Any:
        if value is None:
            return None
        return json.loads(value)

    def record(self, event: EventRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO events (
                event_id, schema_version, timestamp, session_id, run_id, event_type,
                request_name, model, temperature, max_tokens, messages, raw_output,
                status_code, latency_ms, token_usage, parsed_output, validation_errors,
                action, action_payload, tool_name, tool_arguments, tool_result, tool_call_id,
                agent, iteration, agent_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.schema_version,
                event.timestamp,
                event.session_id,
                event.run_id,
                event.event_type,
                event.request_name,
                event.model,
                event.temperature,
                event.max_tokens,
                self._dumps(event.messages),
                event.raw_output,
                event.status_code,
                event.latency_ms,
                self._dumps(event.token_usage),
                self._dumps(event.parsed_output),
                self._dumps(event.validation_errors),
                event.action,
                self._dumps(event.action_payload),
                event.tool_name,
                self._dumps(event.tool_arguments),
                self._dumps(event.tool_result),
                event.tool_call_id,
                event.agent,
                event.iteration,
                event.agent_round,
            ),
        )
        self._connection.commit()

    def query(
        self,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)

        query = "SELECT * FROM events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))

        rows = self._connection.execute(query, tuple(params)).fetchall()
        events: list[EventRecord] = []
        for row in rows:
            events.append(
                EventRecord(
                    event_id=row["event_id"],
                    schema_version=row["schema_version"],
                    timestamp=row["timestamp"],
                    session_id=row["session_id"],
                    run_id=row["run_id"],
                    event_type=row["event_type"],
                    request_name=row["request_name"],
                    model=row["model"],
                    temperature=row["temperature"],
                    max_tokens=row["max_tokens"],
                    messages=self._loads(row["messages"]),
                    raw_output=row["raw_output"],
                    status_code=row["status_code"],
                    latency_ms=row["latency_ms"],
                    token_usage=self._loads(row["token_usage"]),
                    parsed_output=self._loads(row["parsed_output"]),
                    validation_errors=self._loads(row["validation_errors"]),
                    action=row["action"],
                    action_payload=self._loads(row["action_payload"]),
                    tool_name=row["tool_name"],
                    tool_arguments=self._loads(row["tool_arguments"]),
                    tool_result=self._loads(row["tool_result"]),
                    tool_call_id=row["tool_call_id"],
                    agent=row["agent"],
                    iteration=row["iteration"],
                    agent_round=row["agent_round"],
                )
            )
        return events

    # ------------------------------------------------------------------
    # MonitoringStore: llm_calls
    # ------------------------------------------------------------------

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO llm_calls (
                id, session_id, run_id, timestamp, model, prompt, output,
                output_code, latency_ms, token_usage, temperature, max_tokens,
                stage, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.run_id,
                record.timestamp,
                record.model,
                self._dumps(record.prompt),
                record.output,
                record.output_code,
                record.latency_ms,
                self._dumps(record.token_usage),
                record.temperature,
                record.max_tokens,
                record.stage,
                record.schema_version,
            ),
        )
        self._connection.commit()

    def query_llm_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[LLMCallRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM llm_calls"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._connection.execute(query, tuple(params)).fetchall()
        return [
            LLMCallRecord(
                id=row["id"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                timestamp=row["timestamp"],
                model=row["model"],
                prompt=self._loads(row["prompt"]),
                output=row["output"],
                output_code=row["output_code"],
                latency_ms=row["latency_ms"],
                token_usage=self._loads(row["token_usage"]),
                temperature=row["temperature"],
                max_tokens=row["max_tokens"],
                stage=row["stage"],
                schema_version=row["schema_version"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # MonitoringStore: tool_calls
    # ------------------------------------------------------------------

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO tool_calls (
                id, session_id, run_id, timestamp, tool_name, tool_call_id,
                arguments, result, agent_round, stage, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.run_id,
                record.timestamp,
                record.tool_name,
                record.tool_call_id,
                self._dumps(record.arguments),
                self._dumps(record.result),
                record.agent_round,
                record.stage,
                record.schema_version,
            ),
        )
        self._connection.commit()

    def query_tool_calls(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM tool_calls"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._connection.execute(query, tuple(params)).fetchall()
        return [
            ToolCallRecord(
                id=row["id"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                timestamp=row["timestamp"],
                tool_name=row["tool_name"],
                tool_call_id=row["tool_call_id"],
                arguments=self._loads(row["arguments"]),
                result=self._loads(row["result"]),
                agent_round=row["agent_round"],
                stage=row["stage"],
                schema_version=row["schema_version"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # MonitoringStore: agent_performance
    # ------------------------------------------------------------------

    def record_agent_performance(self, record: AgentPerformanceRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO agent_performance (
                id, session_id, run_id, timestamp, model,
                total_llm_calls, total_tool_calls, total_iterations,
                total_latency_ms, total_tokens,
                portfolio_return_1y, portfolio_current, portfolio_min, portfolio_max,
                tickers, weights, status, error_message, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                total_llm_calls=excluded.total_llm_calls,
                total_tool_calls=excluded.total_tool_calls,
                total_iterations=excluded.total_iterations,
                total_latency_ms=excluded.total_latency_ms,
                total_tokens=excluded.total_tokens,
                portfolio_return_1y=excluded.portfolio_return_1y,
                portfolio_current=excluded.portfolio_current,
                portfolio_min=excluded.portfolio_min,
                portfolio_max=excluded.portfolio_max,
                tickers=excluded.tickers,
                weights=excluded.weights,
                status=excluded.status,
                error_message=excluded.error_message,
                timestamp=excluded.timestamp
            """,
            (
                record.id,
                record.session_id,
                record.run_id,
                record.timestamp,
                record.model,
                record.total_llm_calls,
                record.total_tool_calls,
                record.total_iterations,
                record.total_latency_ms,
                record.total_tokens,
                record.portfolio_return_1y,
                record.portfolio_current,
                record.portfolio_min,
                record.portfolio_max,
                self._dumps(record.tickers),
                self._dumps(record.weights),
                record.status,
                record.error_message,
                record.schema_version,
            ),
        )
        self._connection.commit()

    def query_agent_performance(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentPerformanceRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM agent_performance"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._connection.execute(query, tuple(params)).fetchall()
        return [
            AgentPerformanceRecord(
                id=row["id"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                timestamp=row["timestamp"],
                model=row["model"],
                total_llm_calls=row["total_llm_calls"],
                total_tool_calls=row["total_tool_calls"],
                total_iterations=row["total_iterations"],
                total_latency_ms=row["total_latency_ms"],
                total_tokens=row["total_tokens"],
                portfolio_return_1y=row["portfolio_return_1y"],
                portfolio_current=row["portfolio_current"],
                portfolio_min=row["portfolio_min"],
                portfolio_max=row["portfolio_max"],
                tickers=self._loads(row["tickers"]) or [],
                weights=self._loads(row["weights"]) or {},
                status=row["status"],
                error_message=row["error_message"],
                schema_version=row["schema_version"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()
