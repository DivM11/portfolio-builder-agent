"""Event models used by storage backends and instrumentation points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class EventRecord:
    event_type: str
    session_id: str
    run_id: str | None = None
    request_name: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    messages: list[dict[str, Any]] | None = None
    raw_output: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None
    parsed_output: dict[str, Any] | None = None
    validation_errors: list[str] | None = None
    action: str | None = None
    action_payload: dict[str, Any] | None = None
    agent: str | None = None
    iteration: int | None = None
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
