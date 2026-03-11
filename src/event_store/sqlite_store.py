"""SQLite-backed event store implementation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.event_store.models import EventRecord


class SQLiteEventStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
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
                agent TEXT,
                iteration INTEGER
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")
        self._connection.commit()

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
                action, action_payload, agent, iteration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                event.agent,
                event.iteration,
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
                    agent=row["agent"],
                    iteration=row["iteration"],
                )
            )
        return events

    def close(self) -> None:
        self._connection.close()
