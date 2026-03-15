"""Agent context and result models for the single tool-based agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentContext:
    user_input: str
    portfolio_size: float
    excluded_tickers: tuple[str, ...] = ()
    session_id: str | None = None
    run_id: str | None = None


@dataclass
class AgentResult:
    tickers: list[str] = field(default_factory=list)
    data_by_ticker: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary_text: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    allocation: dict[str, float] = field(default_factory=dict)
    analysis_text: str = ""
    suggestions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
