"""Centralized logging configuration for app and services."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TextIO

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="n/a")
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="n/a")


class CorrelationFilter(logging.Filter):
    """Inject session/run IDs into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        record.run_id = run_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Emit compact JSON logs for container stdout."""

    def __init__(self, include_timestamps: bool = True) -> None:
        super().__init__()
        self.include_timestamps = include_timestamps

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", "n/a"),
            "run_id": getattr(record, "run_id", "n/a"),
        }
        if self.include_timestamps:
            payload["timestamp"] = datetime.now(UTC).isoformat(timespec="milliseconds")
        return json.dumps(payload, separators=(",", ":"))


def set_log_context(*, session_id: str | None = None, run_id: str | None = None) -> None:
    """Set correlation IDs for downstream log lines in this execution context."""

    if session_id is not None:
        session_id_var.set(session_id)
    if run_id is not None:
        run_id_var.set(run_id)


def configure_logging(config: Mapping[str, Any] | None = None, *, stream: TextIO | None = None) -> None:
    """Configure root logging handlers and format from runtime config."""

    cfg = dict(config or {})
    level_name = str(cfg.get("level", "INFO")).upper()
    log_format = str(cfg.get("format", "json")).lower()
    include_timestamps = bool(cfg.get("include_timestamps", True))
    include_ids = bool(cfg.get("include_correlation_ids", True))

    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(stream or sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter(include_timestamps=include_timestamps))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s [session=%(session_id)s run=%(run_id)s] %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    if include_ids:
        handler.addFilter(CorrelationFilter())
    root.addHandler(handler)
