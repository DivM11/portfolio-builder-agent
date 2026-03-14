"""Summary tool used by the portfolio agent."""

from __future__ import annotations

from typing import Any

from src.tickr_data_manager import TickrDataManager
from src.tickr_summary_manager import TickrSummaryManager


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "build_summary",
            "description": "Build a compact summary for a set of tickers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers included in summary.",
                    }
                },
                "required": ["tickers"],
            },
        },
    }


def build_summary_tool(
    arguments: dict[str, Any],
    *,
    tickr_data_manager: TickrDataManager,
    tickr_summary_manager: TickrSummaryManager,
    financial_metrics: list[str],
) -> dict[str, Any]:
    tickers = [str(t).upper() for t in arguments.get("tickers", []) if str(t).strip()]
    data_by_ticker = tickr_data_manager.get_data_by_ticker(tickers)
    available = [ticker for ticker in tickers if ticker in data_by_ticker]
    summary = tickr_summary_manager.build_or_get_summary(
        tickers=available,
        data_by_ticker=data_by_ticker,
        financial_metrics=financial_metrics,
        data_version=tickr_data_manager.cache_version,
    )
    return {
        "summary": summary,
        "ticker_count": len(available),
        "tickers": available,
    }
