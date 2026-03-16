"""Unit tests for ticker summary manager cache behavior."""

import pandas as pd

from src.tickr_summary_manager import TickrSummaryManager


def test_summary_manager_uses_cache_per_version():
    manager = TickrSummaryManager()
    data = {
        "AAPL": {"history": pd.DataFrame({"Close": [1.0, 2.0]})}
    }

    first = manager.build_or_get_summary(
        tickers=["AAPL"],
        data_by_ticker=data,
        data_version=1,
    )
    second = manager.build_or_get_summary(
        tickers=["AAPL"],
        data_by_ticker=data,
        data_version=1,
    )
    refreshed = manager.build_or_get_summary(
        tickers=["AAPL"],
        data_by_ticker=data,
        data_version=2,
    )

    assert first == second
    assert len(manager.cache) == 2
    assert refreshed == first
