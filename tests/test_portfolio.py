"""Unit tests for portfolio allocation helpers."""

from src.portfolio import (
    allocate_portfolio,
    allocate_portfolio_by_weights,
    format_portfolio_allocation,
    normalize_weights,
)


def test_allocate_portfolio_equal_weights():
    allocation = allocate_portfolio(["NVDA", "AMD", "GOOG"], total_amount=500.0)

    assert set(allocation.keys()) == {"NVDA", "AMD", "GOOG"}
    assert round(sum(allocation.values()), 2) == 500.0


def test_format_portfolio_allocation():
    allocation = {"NVDA": 200.0, "AMD": 100.0, "GOOG": 200.0}

    text = format_portfolio_allocation(allocation)

    assert "NVDA" in text
    assert "$200.00" in text


def test_normalize_weights_missing_values():
    weights = {"NVDA": 2.0}
    normalized = normalize_weights(weights, ["NVDA", "AMD"])

    assert round(normalized["NVDA"], 2) == 1.0
    assert round(normalized["AMD"], 2) == 0.0


def test_allocate_portfolio_by_weights():
    allocation = allocate_portfolio_by_weights(
        ["NVDA", "AMD", "GOOG"],
        total_amount=500.0,
        weights={"NVDA": 0.4, "AMD": 0.2, "GOOG": 0.4},
    )

    assert round(sum(allocation.values()), 2) == 500.0
    assert allocation["NVDA"] >= allocation["AMD"]


def test_normalize_weights_empty_tickers_returns_empty() -> None:
    assert normalize_weights({"AAPL": 1.0}, []) == {}


def test_normalize_weights_non_positive_total_falls_back_to_equal() -> None:
    normalized = normalize_weights({"AAPL": -1.0, "MSFT": 0.0}, ["AAPL", "MSFT"])

    assert normalized == {"AAPL": 0.5, "MSFT": 0.5}


def test_allocate_portfolio_by_weights_non_positive_total_returns_empty() -> None:
    assert allocate_portfolio_by_weights(["AAPL"], total_amount=0.0, weights={"AAPL": 1.0}) == {}


def test_format_portfolio_allocation_empty() -> None:
    assert format_portfolio_allocation({}) == "Recommended Portfolio: (none)"
