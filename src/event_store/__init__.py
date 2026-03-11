"""Factory and exports for event store backends."""

from __future__ import annotations

import os
from typing import Any, Mapping

from src.event_store.base import EventStore, NullEventStore
from src.event_store.buffer import BufferedEventStore
from src.event_store.postgres_store import PostgresEventStore
from src.event_store.sqlite_store import SQLiteEventStore


def create_event_store(config: Mapping[str, Any] | None) -> EventStore:
    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return NullEventStore()

    backend = str(cfg.get("backend", "sqlite")).lower()
    if backend == "sqlite":
        sqlite_cfg = dict(cfg.get("sqlite", {}))
        store: EventStore = SQLiteEventStore(sqlite_cfg.get("db_path", "data/events.db"))
    elif backend == "postgres":
        postgres_cfg = dict(cfg.get("postgres", {}))
        dsn = postgres_cfg.get("dsn")
        dsn_env_var = postgres_cfg.get("dsn_env_var")
        if not dsn and dsn_env_var:
            dsn = os.getenv(dsn_env_var)
        if not dsn:
            raise ValueError("event_store.postgres.dsn or dsn_env_var must be configured for postgres backend")
        store = PostgresEventStore(dsn)
    else:
        raise ValueError(f"Unsupported event store backend: {backend}")

    buffer_cfg = dict(cfg.get("buffer", {}))
    if buffer_cfg.get("enabled", False):
        store = BufferedEventStore(
            store,
            flush_interval_seconds=float(buffer_cfg.get("flush_interval_seconds", 5)),
            max_buffer_size=int(buffer_cfg.get("max_buffer_size", 100)),
        )
    return store


__all__ = ["EventStore", "NullEventStore", "create_event_store"]
