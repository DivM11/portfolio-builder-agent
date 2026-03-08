"""Unit tests for agent package exports."""

from src.agents import (
    AgentOrchestrator,
    AgentResult,
    BaseAgent,
    OrchestratorState,
    PortfolioCreatorAgent,
    PortfolioEvaluatorAgent,
)


def test_agents_package_exports():
    assert AgentResult is not None
    assert BaseAgent is not None
    assert PortfolioCreatorAgent is not None
    assert PortfolioEvaluatorAgent is not None
    assert AgentOrchestrator is not None
    assert OrchestratorState is not None
