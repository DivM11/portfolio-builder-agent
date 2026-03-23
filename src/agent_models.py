"""Agent context and result models for the single tool-based agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentContext:
    """Immutable snapshot of a single run's input parameters."""

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


class Context:
    """Mutable context retained across tool calls and user interactions.

    Encapsulates conversation history, intermediate work state, and run
    parameters so the agent can update and carry context throughout its
    entire lifecycle.
    """

    def __init__(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        excluded_tickers: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.user_input = user_input
        self.portfolio_size = portfolio_size
        self.excluded_tickers: list[str] = list(excluded_tickers or [])
        self.session_id = session_id
        self.run_id = run_id
        self.messages: list[dict[str, Any]] = []
        self.work_state: dict[str, Any] = {}
        self.last_result: AgentResult | None = None
        self.tool_invocations: list[dict[str, Any]] = []

    # -- message helpers ---------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def add_assistant_tool_calls_message(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})

    def add_tool_result_message(self, tool_call_id: str, name: str, payload: dict[str, Any]) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": json.dumps(payload),
            }
        )

    # -- state helpers -----------------------------------------------------

    def update_work_state(self, **kwargs: Any) -> None:
        self.work_state.update(kwargs)

    def record_tool_invocation(self, name: str, arguments: dict[str, Any]) -> None:
        self.tool_invocations.append({"name": name, "arguments": arguments})

    def init_work_state(self, seed_result: AgentResult | None = None) -> None:
        if seed_result:
            self.work_state = {
                "tickers": list(seed_result.tickers),
                "weights": dict(seed_result.weights),
                "allocation": dict(seed_result.allocation),
                "summary": seed_result.summary_text,
                "analysis": {},
                "tool_invocations": [],
                "reasoning_text": "",
            }
        else:
            self.work_state = {
                "tickers": [],
                "weights": {},
                "allocation": {},
                "summary": "",
                "analysis": {},
                "tool_invocations": [],
                "reasoning_text": "",
            }

    # -- lifecycle helpers -------------------------------------------------

    def prepare_for_run(self, system_prompt: str) -> None:
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.user_input},
        ]
        self.init_work_state()

    def prepare_for_refine(self, feedback: str, previous: AgentResult) -> None:
        self.user_input = feedback
        self.portfolio_size = float(sum(previous.allocation.values())) if previous.allocation else 0.0
        self.excluded_tickers = list(previous.metadata.get("excluded_tickers", []))
        self.add_message("user", feedback)
        self.init_work_state(seed_result=previous)

    # -- snapshot ----------------------------------------------------------

    def to_agent_context(self) -> AgentContext:
        return AgentContext(
            user_input=self.user_input,
            portfolio_size=self.portfolio_size,
            excluded_tickers=tuple(self.excluded_tickers),
            session_id=self.session_id,
            run_id=self.run_id,
        )
