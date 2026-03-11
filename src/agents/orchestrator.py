"""Orchestration for creator/evaluator portfolio agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from src.agents.base import AgentResult
from src.event_store.base import EventStore, NullEventStore
from src.event_store.models import EventRecord

STATUS_AWAITING_APPROVAL = "AWAITING_APPROVAL"
STATUS_COMPLETE = "COMPLETE"
STATUS_MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    user_input: str
    portfolio_size: float
    iteration: int = 1
    status: str = STATUS_COMPLETE
    creator_result: AgentResult = field(default_factory=AgentResult)
    evaluator_result: AgentResult = field(default_factory=AgentResult)
    pending_suggestions: Dict[str, Any] = field(default_factory=dict)
    selected_tickers: list[str] = field(default_factory=list)
    recommended_tickers: list[str] = field(default_factory=list)
    excluded_tickers: list[str] = field(default_factory=list)


class AgentOrchestrator:
    def __init__(
        self,
        creator_agent: Any,
        evaluator_agent: Any,
        max_iterations: int = 3,
        event_store: EventStore | None = None,
        schema_version: int = 1,
    ) -> None:
        self.creator_agent = creator_agent
        self.evaluator_agent = evaluator_agent
        self.max_iterations = max(1, int(max_iterations))
        self.event_store = event_store or NullEventStore()
        self.schema_version = schema_version

    def _record_user_action(
        self,
        *,
        action: str,
        user_input: str,
        session_id: str | None,
        run_id: str | None,
        iteration: int,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self.event_store.record(
            EventRecord(
                event_type="user_action",
                schema_version=self.schema_version,
                session_id=session_id or "n/a",
                run_id=run_id or "n/a",
                action=action,
                action_payload=payload or {"user_input": user_input},
                agent="orchestrator",
                iteration=iteration,
            )
        )

    @staticmethod
    def _has_actionable_suggestions(suggestions: Dict[str, Any]) -> bool:
        add = suggestions.get("add", []) if isinstance(suggestions, dict) else []
        remove = suggestions.get("remove", []) if isinstance(suggestions, dict) else []
        reweight = suggestions.get("reweight", {}) if isinstance(suggestions, dict) else {}
        return bool(add or remove or reweight)

    def start(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> OrchestratorState:
        logger.info("Orchestrator start iteration=1")
        creator_context = {
            "user_input": user_input,
            "portfolio_size": portfolio_size,
            "session_id": session_id,
            "run_id": run_id,
        }
        creator_result = self.creator_agent.run_initial(creator_context)

        evaluator_context = {
            "user_input": user_input,
            "portfolio_size": portfolio_size,
            "tickers": creator_result.tickers,
            "weights": creator_result.weights,
            "allocation": creator_result.allocation,
            "summary_text": creator_result.summary_text,
            "session_id": session_id,
            "run_id": run_id,
        }
        evaluator_result = self.evaluator_agent.run_initial(evaluator_context)

        suggestions = evaluator_result.suggestions if isinstance(evaluator_result.suggestions, dict) else {}
        if self._has_actionable_suggestions(suggestions):
            status = STATUS_AWAITING_APPROVAL
        else:
            status = STATUS_COMPLETE

        self._record_user_action(
            action="new_prompt",
            user_input=user_input,
            session_id=session_id,
            run_id=run_id,
            iteration=1,
            payload={"user_input": user_input},
        )

        return OrchestratorState(
            user_input=user_input,
            portfolio_size=portfolio_size,
            iteration=1,
            status=status,
            creator_result=creator_result,
            evaluator_result=evaluator_result,
            pending_suggestions=suggestions,
            selected_tickers=list(creator_result.tickers),
            recommended_tickers=list(creator_result.metadata.get("recommended_tickers", [])),
            excluded_tickers=list(creator_result.metadata.get("excluded_tickers", [])),
        )

    def apply_changes(
        self,
        state: OrchestratorState,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> OrchestratorState:
        if state.iteration >= self.max_iterations:
            state.status = STATUS_MAX_ITERATIONS_REACHED
            logger.info("Orchestrator max iterations reached iteration=%s", state.iteration)
            return state

        feedback = state.pending_suggestions or {}
        logger.info("Orchestrator apply_changes iteration=%s", state.iteration)
        creator_context = {
            "user_input": state.user_input,
            "portfolio_size": state.portfolio_size,
            "session_id": session_id,
            "run_id": run_id,
        }
        creator_result = self.creator_agent.run_followup(creator_context, feedback)

        evaluator_context = {
            "user_input": state.user_input,
            "portfolio_size": state.portfolio_size,
            "tickers": creator_result.tickers,
            "weights": creator_result.weights,
            "allocation": creator_result.allocation,
            "summary_text": creator_result.summary_text,
            "previous_analysis": state.evaluator_result.analysis_text,
            "session_id": session_id,
            "run_id": run_id,
        }
        evaluator_result = self.evaluator_agent.run_followup(evaluator_context, feedback)

        state.creator_result = creator_result
        state.evaluator_result = evaluator_result
        state.pending_suggestions = evaluator_result.suggestions if isinstance(evaluator_result.suggestions, dict) else {}
        state.selected_tickers = list(creator_result.tickers)
        state.recommended_tickers = list(creator_result.metadata.get("recommended_tickers", []))
        state.excluded_tickers = list(creator_result.metadata.get("excluded_tickers", []))
        state.iteration += 1

        if state.iteration >= self.max_iterations and self._has_actionable_suggestions(state.pending_suggestions):
            state.status = STATUS_MAX_ITERATIONS_REACHED
        elif self._has_actionable_suggestions(state.pending_suggestions):
            state.status = STATUS_AWAITING_APPROVAL
        else:
            state.status = STATUS_COMPLETE

        self._record_user_action(
            action="accept_changes",
            user_input=state.user_input,
            session_id=session_id,
            run_id=run_id,
            iteration=state.iteration,
            payload=feedback,
        )

        return state

    def reject_changes(
        self,
        state: OrchestratorState,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> OrchestratorState:
        state.pending_suggestions = {}
        state.status = STATUS_COMPLETE
        logger.info("Orchestrator reject_changes iteration=%s", state.iteration)
        self._record_user_action(
            action="reject_changes",
            user_input=state.user_input,
            session_id=session_id,
            run_id=run_id,
            iteration=state.iteration,
            payload={},
        )
        return state
