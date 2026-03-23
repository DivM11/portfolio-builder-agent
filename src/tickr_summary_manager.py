"""Ticker summary caching keyed by manager version and ticker set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.summaries import build_portfolio_summary


@dataclass
class TickrSummaryManager:
    cache: dict[tuple[tuple[str, ...], int], str] = field(default_factory=dict)

    def build_or_get_summary(
        self,
        *,
        tickers: list[str],
        data_by_ticker: dict[str, dict[str, Any]],
        data_version: int,
    ) -> str:
        key = (tuple(sorted(tickers)), data_version)
        if key in self.cache:
            return self.cache[key]

        summary = build_portfolio_summary(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
        )
        self.cache[key] = summary
        return summary
