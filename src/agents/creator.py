"""Portfolio creator agent."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.data_client import create_massive_client, fetch_stock_data
from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_weights_payload,
)
from src.portfolio import allocate_portfolio_by_weights, normalize_weights
from src.summaries import build_portfolio_summary


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
    ) -> None:
        super().__init__(llm_service, config)
        self._massive_client_factory = massive_client_factory
        self._stock_data_fetcher = stock_data_fetcher

    def _get_recommended_tickers(
        self,
        user_query: str,
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

        ticker_system_template = prompts_cfg.get("ticker_system", self.DEFAULT_TICKER_SYSTEM)
        ticker_template = prompts_cfg.get("ticker_template", self.DEFAULT_TICKER_TEMPLATE)

        if followup:
            system_prompt = prompts_cfg.get("creator_followup_system", ticker_system_template).format(
                num_tickers=max_tickers
            )
            prompt = prompts_cfg.get("creator_followup_template", ticker_template).format(
                user_input=user_query,
                num_tickers=max_tickers,
            )
            request_name = "ticker_generation_followup"
        else:
            system_prompt = ticker_system_template.format(num_tickers=max_tickers)
            prompt = ticker_template.format(user_input=user_query, num_tickers=max_tickers)
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
    ) -> tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, List[str]]]:
        stocks_cfg = self.config["stocks"]
        massive_api_key = self.config.get("massive", {}).get("api", {}).get("api_key")
        if not massive_api_key:
            raise ValueError("Missing Massive.com API key.")

        massive_client = self._massive_client_factory(massive_api_key)
        data_by_ticker: Dict[str, Dict[str, Any]] = {}
        tickers_with_history: List[str] = []
        failed_history_by_status: Dict[str, List[str]] = {
            "rate_limited": [],
            "not_found": [],
            "empty_data": [],
            "unexpected_error": [],
        }

        for ticker in tickers:
            try:
                ticker_data = self._stock_data_fetcher(
                    ticker=ticker,
                    history_period=stocks_cfg.get("history_period", "1y"),
                    financials_period=stocks_cfg.get("financials_period", "quarterly"),
                    massive_client=massive_client,
                )
            except Exception:
                failed_history_by_status["unexpected_error"].append(ticker)
                continue

            history = ticker_data.get("history", pd.DataFrame())
            history_status = ticker_data.get("history_status", "ok")
            if history is None or history.empty or "Close" not in history.columns:
                if history_status not in failed_history_by_status:
                    history_status = "unexpected_error"
                if history_status == "ok":
                    history_status = "empty_data"
                failed_history_by_status[history_status].append(ticker)
                continue

            data_by_ticker[ticker] = ticker_data
            tickers_with_history.append(ticker)

        return data_by_ticker, tickers_with_history, failed_history_by_status

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
        dropped = sorted(set(tickers) - set(parsed.keys())) if parsed else []
        weights = normalize_weights(parsed, tickers) if parsed else normalize_weights({}, tickers)
        return weights, {
            "weights_raw_text": text,
            "weights_parse_failed": not bool(parsed),
            "weights_dropped": dropped,
        }

    def _run(
        self,
        context: Dict[str, Any],
        *,
        followup: bool,
    ) -> AgentResult:
        user_input = context["user_input"]
        portfolio_size = float(context["portfolio_size"])
        stocks_cfg = self.config.get("stocks", {})
        max_tickers = int(stocks_cfg.get("max_tickers", 10))
        session_id = context.get("session_id")
        run_id = context.get("run_id")

        recommended_tickers, ticker_raw_output = self._get_recommended_tickers(
            user_query=user_input,
            max_tickers=max_tickers,
            session_id=session_id,
            run_id=run_id,
            followup=followup,
        )
        merged_tickers = recommended_tickers[:max_tickers] if max_tickers > 0 else []

        if not has_valid_tickers(merged_tickers):
            raise ValueError("No valid ticker symbols found from LLM output.")

        data_by_ticker, tickers_with_history, failed_history = self._fetch_data_and_filter_tickers(merged_tickers)
        if not tickers_with_history:
            raise ValueError("Could not fetch historical price data for any suggested ticker.")

        summary_text = build_portfolio_summary(
            tickers=tickers_with_history,
            data_by_ticker=data_by_ticker,
            financial_metrics=stocks_cfg.get("financials_metrics", []),
        )
        weights, weights_meta = self._generate_weights(
            user_input=user_input,
            tickers=tickers_with_history,
            summary_text=summary_text,
            session_id=session_id,
            run_id=run_id,
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
                "ticker_raw_output": ticker_raw_output,
                "failed_history_by_status": failed_history,
                **weights_meta,
            },
        )

    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        return self._run(context, followup=False)

    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        _ = feedback
        return self._run(context, followup=True)
