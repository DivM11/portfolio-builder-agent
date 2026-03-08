"""Prompt input/output validation strategies and runtime runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from src.llm_validation import has_valid_tickers, validate_weight_sum


class PromptValidationError(ValueError):
    """Raised when runtime prompt validation fails in fail-fast mode."""


class OutputValidationStrategy(ABC):
    """Abstract strategy for validating prompt output with optional input checks."""

    stage: str

    def validate_input(self, payload: Dict[str, Any]) -> List[str]:
        return []

    @abstractmethod
    def validate_output(self, payload: Dict[str, Any]) -> List[str]:
        raise NotImplementedError


class TickerPromptValidator(OutputValidationStrategy):
    stage = "ticker"

    def validate_input(self, payload: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not str(payload.get("user_query", "")).strip():
            errors.append("User query must be non-empty.")
        if int(payload.get("max_tickers", 0)) <= 0:
            errors.append("max_tickers must be greater than 0.")
        return errors

    def validate_output(self, payload: Dict[str, Any]) -> List[str]:
        parsed = payload.get("parsed_output", [])
        if not has_valid_tickers(parsed):
            return ["Ticker output contains no valid symbols."]
        return []


class PortfolioPromptValidator(OutputValidationStrategy):
    stage = "portfolio"

    def validate_input(self, payload: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not str(payload.get("user_input", "")).strip():
            errors.append("Portfolio prompt requires user_input.")
        tickers = payload.get("tickers", [])
        if not tickers:
            errors.append("Portfolio prompt requires at least one ticker.")
        if not str(payload.get("summary_text", "")).strip():
            errors.append("Portfolio prompt requires summary_text.")
        return errors

    def validate_output(self, payload: Dict[str, Any]) -> List[str]:
        parsed = payload.get("parsed_output", {})
        if not isinstance(parsed, dict) or not parsed:
            return ["Portfolio output must provide weights."]

        is_valid_sum, total = validate_weight_sum(parsed)
        if not is_valid_sum:
            return [f"Portfolio weights must sum to 1.0 (+/- tolerance). Found {total:.4f}."]
        return []


class AnalysisPromptValidator(OutputValidationStrategy):
    stage = "analysis"

    def validate_input(self, payload: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not str(payload.get("user_input", "")).strip():
            errors.append("Analysis prompt requires user_input.")
        if not payload.get("tickers", []):
            errors.append("Analysis prompt requires tickers.")
        if not str(payload.get("summary_text", "")).strip():
            errors.append("Analysis prompt requires summary_text.")
        return errors

    def validate_output(self, payload: Dict[str, Any]) -> List[str]:
        raw_output = str(payload.get("raw_output", "")).strip()
        if not raw_output:
            return ["Analysis output must be non-empty."]
        return []


@dataclass(frozen=True)
class PromptValidationConfig:
    enabled: bool = True
    validate_input: bool = True
    validate_output: bool = True
    fail_fast: bool = False
    prompts: Dict[str, bool] | None = None

    @classmethod
    def from_config(cls, config: Dict[str, Any] | None) -> "PromptValidationConfig":
        cfg = config or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            validate_input=bool(cfg.get("validate_input", True)),
            validate_output=bool(cfg.get("validate_output", True)),
            fail_fast=bool(cfg.get("fail_fast", False)),
            prompts=cfg.get("prompts", {}),
        )


class PromptValidationRunner:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = PromptValidationConfig.from_config(config)

    def _is_stage_enabled(self, stage: str) -> bool:
        stage_map = self.config.prompts or {}
        return bool(stage_map.get(stage, True))

    def _handle_errors(self, phase: str, stage: str, errors: List[str]) -> List[str]:
        tagged = [f"{stage}.{phase}: {err}" for err in errors]
        if tagged and self.config.fail_fast:
            raise PromptValidationError("; ".join(tagged))
        return tagged

    def validate_input(
        self,
        stage: str,
        validator: OutputValidationStrategy,
        payload: Dict[str, Any],
    ) -> List[str]:
        if not self.config.enabled or not self.config.validate_input or not self._is_stage_enabled(stage):
            return []
        return self._handle_errors("input", stage, validator.validate_input(payload))

    def validate_output(
        self,
        stage: str,
        validator: OutputValidationStrategy,
        payload: Dict[str, Any],
    ) -> List[str]:
        if not self.config.enabled or not self.config.validate_output or not self._is_stage_enabled(stage):
            return []
        return self._handle_errors("output", stage, validator.validate_output(payload))
