"""Portfolio analytics tool used by the portfolio agent."""

from __future__ import annotations

from typing import Any

from src.summaries import (
    build_portfolio_returns_series,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)
from src.tickr_data_manager import TickrDataManager


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "analyze_portfolio",
            "description": "Compute portfolio performance and financial aggregates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers in the portfolio.",
                    },
                    "weights": {
                        "type": "object",
                        "description": "Ticker-to-weight map.",
                    },
                },
                "required": ["tickers", "weights"],
            },
        },
    }


def analyze_portfolio_tool(
    arguments: dict[str, Any],
    *,
    tickr_data_manager: TickrDataManager,
    financial_metrics: list[str],
) -> dict[str, Any]:
    tickers = [str(t).upper() for t in arguments.get("tickers", []) if str(t).strip()]
    weights_raw = arguments.get("weights", {})
    if not isinstance(weights_raw, dict):
        weights_raw = {}
    weights = {
        str(ticker).upper(): float(value)
        for ticker, value in weights_raw.items()
        if str(ticker).strip()
    }

    data_by_ticker = tickr_data_manager.get_data_by_ticker(tickers)
    returns_series = build_portfolio_returns_series(
        {ticker: payload.get("history") for ticker, payload in data_by_ticker.items()},
        weights,
    )
    stats = summarize_portfolio_stats(returns_series)
    financials = summarize_portfolio_financials(
        {ticker: payload.get("financials") for ticker, payload in data_by_ticker.items()},
        weights,
        financial_metrics,
    )
    return {
        "stats": stats,
        "financials": financials,
        "returns_data_points": int(len(returns_series)),
    }
