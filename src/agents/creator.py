"""Portfolio creator agent."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.agents.base import AgentResult, BaseAgent
from src.data_client import create_massive_client, fetch_stock_data
from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_weights_payload,
)
from src.agent_models import CreatorContext, CreatorPrompts
from src.event_store.models import EventRecord
from src.portfolio import allocate_portfolio_by_weights, normalize_weights
from src.prompt_validation import (
    PortfolioPromptValidator,
    PromptValidationRunner,
    TickerPromptValidator,
)
from src.tickr_data_manager import TickrDataManager, ProgressCallback
from src.tickr_summary_manager import TickrSummaryManager

TickerCallback = Callable[[List[str]], None]


logger = logging.getLogger(__name__)


class PortfolioCreatorAgent(BaseAgent):
    DEFAULT_TICKER_SYSTEM = (
        "You are a financial assistant. Return up to {num_tickers} comma-separated US stock tickers. "
        "No explanations, no commentary."
    )
    DEFAULT_TICKER_TEMPLATE = "Create a portfolio with up to {num_tickers} US stock tickers for: {user_input}"
    DEFAULT_WEIGHTS_SYSTEM = (
        "You are a portfolio allocator. Respond with a single JSON object only. "
        "No markdown fences, no explanation, no code blocks."
    )
    DEFAULT_WEIGHTS_TEMPLATE = (
        "Allocate portfolio weights that sum to exactly 1.0 across these tickers.\n"
        "Return ONLY a JSON object: {\"weights\": {\"TICKER\": 0.25}}\n\n"
        "User preferences: {user_input}\nTickers: {tickers}\nSummary:\n{summary}"
    )
    def __init__(
        self,
        llm_service: Any,
        config: Dict[str, Any],
        *,
        massive_client_factory: Callable[[str], Any] = create_massive_client,
        stock_data_fetcher: Callable[..., Dict[str, Any]] = fetch_stock_data,
        tickr_data_manager: TickrDataManager | None = None,
        tickr_summary_manager: TickrSummaryManager | None = None,
    ) -> None:
        super().__init__(llm_service, config)
        self._massive_client_factory = massive_client_factory
        self._stock_data_fetcher = stock_data_fetcher
        self._validation_runner = PromptValidationRunner(config.get("validations", {}))
        self._ticker_validator = TickerPromptValidator()
        self._portfolio_validator = PortfolioPromptValidator()
        self._tickr_data_manager = tickr_data_manager or TickrDataManager()
        self._tickr_summary_manager = tickr_summary_manager or TickrSummaryManager()

    @staticmethod
    def _apply_feedback_tickers(
        recommended_tickers: List[str],
        add: List[str],
        remove: List[str],
        max_tickers: int,
    ) -> tuple[List[str], List[str]]:
        deduped: List[str] = []
        for ticker in recommended_tickers:
            ticker_up = str(ticker).upper()
            if ticker_up not in deduped:
                deduped.append(ticker_up)

        remove_set = {ticker.upper() for ticker in remove}
        for ticker in add:
            ticker_up = str(ticker).upper()
            if ticker_up not in deduped and ticker_up not in remove_set:
                deduped.append(ticker_up)

        filtered = [ticker for ticker in deduped if ticker not in remove_set]
        if max_tickers > 0:
            filtered = filtered[:max_tickers]
        return filtered, sorted(remove_set)

    def _get_recommended_tickers(
        self,
        context: CreatorContext,
        max_tickers: int,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        followup: bool = False,
    ) -> tuple[List[str], str]:
        openrouter_cfg = self.config.get("openrouter", {})
        prompts_cfg = openrouter_cfg.get("prompts", {})
        outputs_cfg = openrouter_cfg.get("outputs", {})
        temperatures_cfg = openrouter_cfg.get("temperatures", {})
        models_cfg = openrouter_cfg.get("models", {})
        dashboard_cfg = self.config.get("dashboard", {})

        prompts = CreatorPrompts(
            ticker_system=prompts_cfg.get("ticker_system", self.DEFAULT_TICKER_SYSTEM),
            ticker_template=prompts_cfg.get("ticker_template", self.DEFAULT_TICKER_TEMPLATE),
            ticker_followup_system=prompts_cfg.get("creator_followup_system", self.DEFAULT_TICKER_SYSTEM),
            ticker_followup_template=prompts_cfg.get("creator_followup_template", self.DEFAULT_TICKER_TEMPLATE),
            weights_system=prompts_cfg.get("weights_system", self.DEFAULT_WEIGHTS_SYSTEM),
            weights_template=prompts_cfg.get("weights_template", self.DEFAULT_WEIGHTS_TEMPLATE),
        )

        self._validation_runner.validate_input(
            "ticker",
            self._ticker_validator,
            {
                "user_query": context.user_input,
                "max_tickers": max_tickers,
            },
        )

        if followup:
            system_prompt = prompts.ticker_followup_system.format(
                num_tickers=max_tickers
            )
            prompt = prompts.ticker_followup_template.format(
                user_input=context.user_input,
                num_tickers=max_tickers,
            )
            request_name = "ticker_generation_followup"
        else:
            system_prompt = prompts.ticker_system.format(num_tickers=max_tickers)
            prompt = prompts.ticker_template.format(user_input=context.user_input, num_tickers=max_tickers)
            request_name = "ticker_generation"

        response, _ = self.llm_service.complete(
            request_name=request_name,
            model=models_cfg.get("ticker", "anthropic/claude-3.5-haiku"),
            max_tokens=outputs_cfg.get("ticker_max_tokens", 100),
            temperature=temperatures_cfg.get("ticker", 0.2),
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw_output = self.llm_service.extract_message_text(response)
        tickers = extract_valid_tickers(raw_output, delimiter=dashboard_cfg.get("ticker_delimiter", ","))
        return tickers, raw_output

    def _fetch_data_and_filter_tickers(
        self,
        tickers: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, List[str]]]:
        stocks_cfg = self.config["stocks"]
        massive_api_key = self.config.get("massive", {}).get("api", {}).get("api_key")
        if not massive_api_key:
            raise ValueError("Missing Massive.com API key.")

        try:
            massive_client = self._massive_client_factory(massive_api_key)
        except TypeError:
            # Test doubles may still expose a zero-arg factory.
            massive_client = self._massive_client_factory()

        def _fetcher_proxy(**kwargs: Any) -> Dict[str, Any]:
            try:
                return self._stock_data_fetcher(
                    kwargs["ticker"],
                    kwargs["history_period"],
                    kwargs["financials_period"],
                    kwargs["massive_client"],
                )
            except TypeError:
                return self._stock_data_fetcher(**kwargs)

        result = self._tickr_data_manager.fetch_for_tickers(
            tickers=tickers,
            fetcher=_fetcher_proxy,
            history_period=stocks_cfg.get("history_period", "1y"),
            financials_period=stocks_cfg.get("financials_period", "quarterly"),
            massive_client=massive_client,
            progress_callback=progress_callback,
        )

        return result.data_by_ticker, result.tickers_with_history, result.failed_history_by_status

    def _generate_weights(
        self,
        user_input: str,
        tickers: List[str],
        summary_text: str,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        openrouter_cfg = self.config.get("openrouter", {})
        prompts_cfg = openrouter_cfg.get("prompts", {})
        outputs_cfg = openrouter_cfg.get("outputs", {})
        temperatures_cfg = openrouter_cfg.get("temperatures", {})
        models_cfg = openrouter_cfg.get("models", {})

        validation_errors: List[str] = self._validation_runner.validate_input(
            "portfolio",
            self._portfolio_validator,
            {
                "user_input": user_input,
                "tickers": tickers,
                "summary_text": summary_text,
            },
        )

        weights_prompt = prompts_cfg.get("weights_template", self.DEFAULT_WEIGHTS_TEMPLATE).format(
            user_input=user_input,
            tickers=", ".join(tickers),
            summary=summary_text,
        )
        response, _ = self.llm_service.complete(
            request_name="weights_generation",
            model=models_cfg.get("weights", "anthropic/claude-3.5-haiku"),
            max_tokens=outputs_cfg.get("weights_max_tokens", 400),
            temperature=temperatures_cfg.get("weights", 0.1),
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": prompts_cfg.get("weights_system", self.DEFAULT_WEIGHTS_SYSTEM)},
                {"role": "user", "content": weights_prompt},
            ],
        )
        text = self.llm_service.extract_message_text(response)
        parsed = parse_weights_payload(text)
        validation_errors.extend(
            self._validation_runner.validate_output(
                "portfolio",
                self._portfolio_validator,
                {
                    "raw_output": text,
                    "parsed_output": parsed,
                },
            )
        )
        dropped = sorted(set(tickers) - set(parsed.keys())) if parsed else []
        weights = normalize_weights(parsed, tickers) if parsed else normalize_weights({}, tickers)
        return weights, {
            "weights_raw_text": text,
            "weights_parse_failed": not bool(parsed),
            "weights_dropped": dropped,
            "validation_errors": validation_errors,
        }

    def _run(
        self,
        context: Dict[str, Any],
        *,
        followup: bool,
        feedback: Dict[str, Any] | None = None,
    ) -> AgentResult:
        ticker_callback: Optional[TickerCallback] = context.get("_ticker_callback")
        progress_callback: Optional[ProgressCallback] = context.get("_progress_callback")
        creator_context = CreatorContext(
            user_input=context["user_input"],
            portfolio_size=float(context["portfolio_size"]),
            session_id=context.get("session_id"),
            run_id=context.get("run_id"),
        )
        user_input = creator_context.user_input
        portfolio_size = creator_context.portfolio_size
        stocks_cfg = self.config.get("stocks", {})
        max_tickers = int(stocks_cfg.get("max_tickers", 10))
        session_id = creator_context.session_id
        run_id = creator_context.run_id

        recommended_tickers, ticker_raw_output = self._get_recommended_tickers(
            context=creator_context,
            max_tickers=max_tickers,
            session_id=session_id,
            run_id=run_id,
            followup=followup,
        )
        validation_errors: List[str] = self._validation_runner.validate_output(
            "ticker",
            self._ticker_validator,
            {
                "raw_output": ticker_raw_output,
                "parsed_output": recommended_tickers,
                "max_tickers": max_tickers,
            },
        )
        merged_tickers = recommended_tickers[:max_tickers] if max_tickers > 0 else []
        if hasattr(self.llm_service, "event_store"):
            self.llm_service.event_store.record(
                EventRecord(
                    event_type="llm_parsed",
                    schema_version=int(self.config.get("event_store", {}).get("schema_version", 1)),
                    session_id=session_id or "n/a",
                    run_id=run_id or "n/a",
                    request_name="ticker_generation_followup" if followup else "ticker_generation",
                    parsed_output={"tickers": merged_tickers},
                    validation_errors=validation_errors,
                    agent="creator",
                )
            )
        feedback = feedback or {}
        add_tickers = [str(item).upper() for item in feedback.get("add", [])]
        remove_tickers = [str(item).upper() for item in feedback.get("remove", [])]
        merged_tickers, excluded_tickers = self._apply_feedback_tickers(
            merged_tickers,
            add=add_tickers,
            remove=remove_tickers,
            max_tickers=max_tickers,
        )

        if not has_valid_tickers(merged_tickers):
            logger.warning(
                "Ticker validation failed. Raw model output from OpenRouter: %s",
                ticker_raw_output,
            )
            raise ValueError("No valid ticker symbols found from LLM output.")

        if ticker_callback is not None:
            ticker_callback(merged_tickers)

        data_by_ticker, tickers_with_history, failed_history = self._fetch_data_and_filter_tickers(
            merged_tickers, progress_callback=progress_callback,
        )
        if not tickers_with_history:
            rate_limited_tickers = failed_history.get("rate_limited", [])
            if rate_limited_tickers:
                raise ValueError(
                    "Rate-limited while fetching historical price data for: "
                    + ", ".join(rate_limited_tickers)
                )
            raise ValueError("Could not fetch historical price data for any suggested ticker.")

        summary_text = self._tickr_summary_manager.build_or_get_summary(
            tickers=tickers_with_history,
            data_by_ticker=data_by_ticker,
            financial_metrics=stocks_cfg.get("financials_metrics", []),
            data_version=self._tickr_data_manager.cache_version,
        )
        weights, weights_meta = self._generate_weights(
            user_input=user_input,
            tickers=tickers_with_history,
            summary_text=summary_text,
            session_id=session_id,
            run_id=run_id,
        )
        validation_errors.extend(weights_meta.pop("validation_errors", []))
        if hasattr(self.llm_service, "event_store"):
            self.llm_service.event_store.record(
                EventRecord(
                    event_type="llm_parsed",
                    schema_version=int(self.config.get("event_store", {}).get("schema_version", 1)),
                    session_id=session_id or "n/a",
                    run_id=run_id or "n/a",
                    request_name="weights_generation",
                    parsed_output={"weights": weights},
                    validation_errors=validation_errors,
                    agent="creator",
                )
            )
        allocation = allocate_portfolio_by_weights(
            tickers=tickers_with_history,
            total_amount=portfolio_size,
            weights=weights,
        )

        return AgentResult(
            tickers=tickers_with_history,
            data_by_ticker=data_by_ticker,
            summary_text=summary_text,
            weights=weights,
            allocation=allocation,
            metadata={
                "recommended_tickers": recommended_tickers,
                "excluded_tickers": excluded_tickers,
                "ticker_raw_output": ticker_raw_output,
                "failed_history_by_status": failed_history,
                "validation_errors": validation_errors,
                **weights_meta,
            },
        )

    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        return self._run(context, followup=False)

    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        return self._run(context, followup=True, feedback=feedback)
