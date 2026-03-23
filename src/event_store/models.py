"""Event models used by storage backends and instrumentation points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


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
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_call_id: str | None = None
    agent: str | None = None
    iteration: int | None = None
    agent_round: int | None = None
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMCallRecord:
    """One row per LLM HTTP round-trip recorded by LLMService."""

    id: str
    session_id: str
    run_id: str | None
    timestamp: str
    model: str | None
    prompt: list[dict[str, Any]] | None
    output: str | None
    output_code: int | None
    latency_ms: float | None
    token_usage: dict[str, Any] | None
    temperature: float | None
    max_tokens: int | None
    stage: str | None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallRecord:
    """One row per tool invocation recorded by PortfolioAgent."""

    id: str
    session_id: str
    run_id: str | None
    timestamp: str
    tool_name: str
    tool_call_id: str | None
    arguments: dict[str, Any] | None
    result: dict[str, Any] | None
    agent_round: int | None
    stage: str | None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPerformanceRecord:
    """One ETL-materialised row per completed agent run."""

    id: str
    session_id: str
    run_id: str
    timestamp: str
    model: str | None
    total_llm_calls: int
    total_tool_calls: int
    total_iterations: int
    total_latency_ms: float
    total_tokens: int
    portfolio_return_1y: float | None
    portfolio_current: float | None
    portfolio_min: float | None
    portfolio_max: float | None
    tickers: list[str]
    weights: dict[str, float]
    status: str
    error_message: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
