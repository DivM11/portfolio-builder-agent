"""Base abstractions for portfolio agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentResult:
    tickers: List[str] = field(default_factory=list)
    data_by_ticker: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary_text: str = ""
    weights: Dict[str, float] = field(default_factory=dict)
    allocation: Dict[str, float] = field(default_factory=dict)
    analysis_text: str = ""
    suggestions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, llm_service: Any, config: Dict[str, Any]) -> None:
        self.llm_service = llm_service
        self.config = config

    @abstractmethod
    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError

    @abstractmethod
    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError
