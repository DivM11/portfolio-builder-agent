"""Data fetch tool used by the portfolio agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.tickr_data_manager import ProgressCallback, TickrDataManager


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_ticker_data",
            "description": "Fetch historical price and financial statement data for tickers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers to fetch.",
                    }
                },
                "required": ["tickers"],
            },
        },
    }


def fetch_ticker_data_tool(
    arguments: dict[str, Any],
    *,
    tickr_data_manager: TickrDataManager,
    stock_data_fetcher: Callable[..., dict[str, Any]],
    history_period: str,
    massive_client: Any,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    tickers_raw = arguments.get("tickers", [])
    tickers = [str(t).upper() for t in tickers_raw if str(t).strip()]
    before = set(tickers)
    fetch_result = tickr_data_manager.fetch_for_tickers(
        tickers=tickers,
        fetcher=stock_data_fetcher,
        history_period=history_period,
        massive_client=massive_client,
        progress_callback=progress_callback,
    )
    fetched = set(fetch_result.fetched_tickers)
    available = set(fetch_result.tickers_with_history)
    cached = sorted(list((before & available) - fetched))
    failed: dict[str, str] = {}
    for status, symbols in fetch_result.failed_history_by_status.items():
        for symbol in symbols:
            failed[symbol] = status

    return {
        "fetched": sorted(fetch_result.fetched_tickers),
        "cached": cached,
        "failed": failed,
        "available_tickers": sorted(fetch_result.tickers_with_history),
        "failed_history_by_status": fetch_result.failed_history_by_status,
    }
