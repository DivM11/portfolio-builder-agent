"""Ticker data caching and refresh management."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

ProgressCallback = Callable[[int, int, str], None]
logger = logging.getLogger(__name__)


@dataclass
class TickrDataFetchResult:
    data_by_ticker: dict[str, dict[str, Any]]
    tickers_with_history: list[str]
    failed_history_by_status: dict[str, list[str]]
    fetched_tickers: list[str]


@dataclass
class TickrDataManager:
    """Keeps ticker data across iterations and fetches only missing symbols."""

    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_version: int = 0

    def update_ticker(self, ticker: str, payload: dict[str, Any]) -> None:
        self.cache[ticker] = payload

    def has_ticker(self, ticker: str) -> bool:
        return ticker in self.cache

    def get_data_by_ticker(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        return {ticker: self.cache[ticker] for ticker in tickers if ticker in self.cache}

    def fetch_for_tickers(
        self,
        tickers: list[str],
        fetcher: Callable[..., dict[str, Any]],
        *,
        history_period: str,
        massive_client: Any,
        progress_callback: ProgressCallback | None = None,
    ) -> TickrDataFetchResult:
        failed_history_by_status: dict[str, list[str]] = {
            "rate_limited": [],
            "not_found": [],
            "empty_data": [],
            "unexpected_error": [],
        }
        tickers_with_history: list[str] = []
        fetched_tickers: list[str] = []

        for idx, ticker in enumerate(tickers):
            if progress_callback is not None:
                progress_callback(idx, len(tickers), ticker)
            ticker_data: dict[str, Any]
            if self.has_ticker(ticker):
                ticker_data = self.cache[ticker]
            else:
                try:
                    ticker_data = fetcher(
                        ticker=ticker,
                        history_period=history_period,
                        client=massive_client,
                    )
                    self.update_ticker(ticker, ticker_data)
                    fetched_tickers.append(ticker)
                except Exception:
                    logger.exception("Ticker fetch failed for %s", ticker)
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

            tickers_with_history.append(ticker)

        if fetched_tickers:
            self.cache_version += 1

        return TickrDataFetchResult(
            data_by_ticker=self.get_data_by_ticker(tickers_with_history),
            tickers_with_history=tickers_with_history,
            failed_history_by_status=failed_history_by_status,
            fetched_tickers=fetched_tickers,
        )
