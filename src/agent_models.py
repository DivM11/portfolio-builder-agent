"""Agent context and result models for the single tool-based agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreatorContext:
    user_input: str
    portfolio_size: float
    session_id: str | None = None
    run_id: str | None = None
    excluded_tickers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluatorContext:
    user_input: str
    portfolio_size: float
    tickers: tuple[str, ...]
    summary_text: str


@dataclass(frozen=True)
class CreatorPrompts:
    ticker_system: str
    ticker_template: str
    ticker_followup_system: str
    ticker_followup_template: str
    weights_system: str
    weights_template: str


@dataclass(frozen=True)
class EvaluatorPrompts:
    analysis_system: str
    analysis_template: str
    analysis_followup_system: str
    analysis_followup_template: str


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
