"""Unit tests for orchestrator event instrumentation."""

from src.agents.base import AgentResult
from src.agents.orchestrator import AgentOrchestrator
from src.event_store.base import EventStore


class DummyCreator:
    def run_initial(self, context):
        return AgentResult(
            tickers=["AAPL"],
            weights={"AAPL": 1.0},
            allocation={"AAPL": context["portfolio_size"]},
            summary_text="summary",
            metadata={"recommended_tickers": ["AAPL"], "excluded_tickers": []},
        )

    def run_followup(self, context, _feedback):
        return AgentResult(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.5, "MSFT": 0.5},
            allocation={"AAPL": context["portfolio_size"] * 0.5, "MSFT": context["portfolio_size"] * 0.5},
            summary_text="summary2",
            metadata={"recommended_tickers": ["AAPL", "MSFT"], "excluded_tickers": []},
        )


class DummyEvaluator:
    def run_initial(self, _context):
        return AgentResult(analysis_text="analysis", suggestions={"add": ["MSFT"], "remove": [], "reweight": {}})

    def run_followup(self, _context, _feedback):
        return AgentResult(analysis_text="updated", suggestions={"add": [], "remove": [], "reweight": {}})


class RecordingStore(EventStore):
    def __init__(self) -> None:
        self.events = []

    def record(self, event):
        self.events.append(event)

    def query(self, **_kwargs):
        return list(self.events)

    def close(self):
        return None


def test_orchestrator_records_user_actions() -> None:
    store = RecordingStore()
    orchestrator = AgentOrchestrator(DummyCreator(), DummyEvaluator(), max_iterations=3, event_store=store)

    state = orchestrator.start(user_input="growth", portfolio_size=1000.0, session_id="session-1", run_id="run-1")
    state = orchestrator.apply_changes(state, session_id="session-1", run_id="run-2")
    orchestrator.reject_changes(state, session_id="session-1", run_id="run-3")

    actions = [event.action for event in store.events if event.event_type == "user_action"]
    assert actions == ["new_prompt", "accept_changes", "reject_changes"]
