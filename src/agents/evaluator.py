"""Portfolio evaluator agent."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.agents.base import AgentResult, BaseAgent
from src.agent_models import EvaluatorContext, EvaluatorPrompts
from src.event_store.models import EventRecord
from src.llm_validation import parse_evaluator_suggestions
from src.prompt_validation import AnalysisPromptValidator, PromptValidationRunner


class PortfolioEvaluatorAgent(BaseAgent):
    DEFAULT_ANALYSIS_SYSTEM = "You are a financial analyst. Provide a concise evaluation of the recommended tickers."
    DEFAULT_ANALYSIS_TEMPLATE = (
        "User preferences: {user_input}\n"
        "Portfolio size: {portfolio_size}\n"
        "Tickers: {tickers}\n"
        "Weights: {weights}\n"
        "Allocation: {allocation}\n"
        "Summary:\n{summary}\n"
        "Provide a brief evaluation and any risks."
    )

    def __init__(self, llm_service: Any, config: Dict[str, Any]) -> None:
        super().__init__(llm_service, config)
        self._validation_runner = PromptValidationRunner(config.get("validations", {}))
        self._analysis_validator = AnalysisPromptValidator()

    def _evaluate(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        tickers: list[str],
        weights: Dict[str, float],
        allocation: Dict[str, float],
        summary_text: str,
        session_id: str | None = None,
        run_id: str | None = None,
        followup: bool = False,
        previous_analysis: str = "",
        applied_changes: Dict[str, Any] | None = None,
    ) -> AgentResult:
        openrouter_cfg = self.config.get("openrouter", {})
        prompts_cfg = openrouter_cfg.get("prompts", {})
        outputs_cfg = openrouter_cfg.get("outputs", {})
        temperatures_cfg = openrouter_cfg.get("temperatures", {})
        models_cfg = openrouter_cfg.get("default_models", {})

        prompts = EvaluatorPrompts(
            analysis_system=prompts_cfg.get("evaluator_system", prompts_cfg.get("analysis_system", self.DEFAULT_ANALYSIS_SYSTEM)),
            analysis_template=prompts_cfg.get("evaluator_template", prompts_cfg.get("analysis_template", self.DEFAULT_ANALYSIS_TEMPLATE)),
            analysis_followup_system=prompts_cfg.get(
                "evaluator_followup_system",
                prompts_cfg.get("evaluator_system", prompts_cfg.get("analysis_system", self.DEFAULT_ANALYSIS_SYSTEM)),
            ),
            analysis_followup_template=prompts_cfg.get(
                "evaluator_followup_template",
                prompts_cfg.get("evaluator_template", prompts_cfg.get("analysis_template", self.DEFAULT_ANALYSIS_TEMPLATE)),
            ),
        )

        eval_context = EvaluatorContext(
            user_input=user_input,
            portfolio_size=portfolio_size,
            tickers=tuple(tickers),
            summary_text=summary_text,
        )

        if followup:
            system_prompt = prompts.analysis_followup_system
            prompt = prompts.analysis_followup_template.format(
                user_input=eval_context.user_input,
                portfolio_size=eval_context.portfolio_size,
                tickers=", ".join(eval_context.tickers),
                weights=json.dumps(weights),
                allocation=json.dumps(allocation),
                summary=eval_context.summary_text,
                previous_analysis=previous_analysis,
                applied_changes=json.dumps(applied_changes or {}),
            )
            request_name = "evaluation_followup"
        else:
            system_prompt = prompts.analysis_system
            prompt = prompts.analysis_template.format(
                user_input=eval_context.user_input,
                portfolio_size=eval_context.portfolio_size,
                tickers=", ".join(eval_context.tickers),
                weights=json.dumps(weights),
                allocation=json.dumps(allocation),
                summary=eval_context.summary_text,
            )
            request_name = "evaluation"

        validation_errors = self._validation_runner.validate_input(
            "analysis",
            self._analysis_validator,
            {
                "user_input": user_input,
                "tickers": eval_context.tickers,
                "summary_text": eval_context.summary_text,
            },
        )

        response, _ = self.llm_service.complete(
            request_name=request_name,
            model=models_cfg.get("evaluator", models_cfg.get("analysis", "anthropic/claude-3.5-haiku")),
            max_tokens=outputs_cfg.get("evaluator_max_tokens", outputs_cfg.get("analysis_max_tokens", 500)),
            temperature=temperatures_cfg.get("evaluator", temperatures_cfg.get("analysis", 0.2)),
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        analysis_text = self.llm_service.extract_message_text(response)
        suggestions = parse_evaluator_suggestions(analysis_text)
        validation_errors.extend(
            self._validation_runner.validate_output(
                "analysis",
                self._analysis_validator,
                {
                    "raw_output": analysis_text,
                    "parsed_output": suggestions,
                },
            )
        )
        if hasattr(self.llm_service, "event_store"):
            self.llm_service.event_store.record(
                EventRecord(
                    event_type="llm_parsed",
                    schema_version=int(self.config.get("event_store", {}).get("schema_version", 1)),
                    session_id=session_id or "n/a",
                    run_id=run_id or "n/a",
                    request_name=request_name,
                    parsed_output=suggestions,
                    validation_errors=validation_errors,
                    agent="evaluator",
                )
            )
        return AgentResult(
            analysis_text=analysis_text,
            suggestions=suggestions,
            metadata={
                "request_name": request_name,
                "validation_errors": validation_errors,
            },
        )

    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        return self._evaluate(
            user_input=context["user_input"],
            portfolio_size=float(context["portfolio_size"]),
            tickers=context["tickers"],
            weights=context["weights"],
            allocation=context["allocation"],
            summary_text=context["summary_text"],
            session_id=context.get("session_id"),
            run_id=context.get("run_id"),
            followup=False,
        )

    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        return self._evaluate(
            user_input=context["user_input"],
            portfolio_size=float(context["portfolio_size"]),
            tickers=context["tickers"],
            weights=context["weights"],
            allocation=context["allocation"],
            summary_text=context["summary_text"],
            session_id=context.get("session_id"),
            run_id=context.get("run_id"),
            followup=True,
            previous_analysis=context.get("previous_analysis", ""),
            applied_changes=feedback,
        )
