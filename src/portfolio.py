"""Portfolio allocation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_DOWN, Decimal


def normalize_weights(weights: dict[str, float] | None, tickers: Iterable[str]) -> dict[str, float]:
    tickers_list = list(tickers)
    if not tickers_list:
        return {}

    if not weights:
        equal = 1.0 / len(tickers_list)
        return {ticker: equal for ticker in tickers_list}

    cleaned = {ticker: max(0.0, float(weights.get(ticker, 0.0))) for ticker in tickers_list}
    total = sum(cleaned.values())
    if total <= 0:
        equal = 1.0 / len(tickers_list)
        return {ticker: equal for ticker in tickers_list}

    return {ticker: value / total for ticker, value in cleaned.items()}


def allocate_portfolio(
    tickers: Iterable[str],
    total_amount: float,
    precision: int = 2,
) -> dict[str, float]:
    """Allocate a total amount across tickers using equal weights."""
    return allocate_portfolio_by_weights(tickers, total_amount, None, precision)


def allocate_portfolio_by_weights(
    tickers: Iterable[str],
    total_amount: float,
    weights: dict[str, float] | None,
    precision: int = 2,
) -> dict[str, float]:
    tickers_list = list(tickers)
    if not tickers_list or total_amount <= 0:
        return {}

    normalized = normalize_weights(weights, tickers_list)
    quant = Decimal("1") / (Decimal(10) ** precision)
    total = Decimal(str(total_amount))

    raw_allocations = {ticker: Decimal(str(normalized[ticker])) * total for ticker in tickers_list}
    rounded = {ticker: value.quantize(quant, rounding=ROUND_DOWN) for ticker, value in raw_allocations.items()}

    allocated = sum(rounded.values())
    remainder = (total - allocated).quantize(quant)
    if remainder <= Decimal("0"):
        return {ticker: float(value) for ticker, value in rounded.items()}

    fractional = sorted(
        tickers_list,
        key=lambda t: raw_allocations[t] - rounded[t],
        reverse=True,
    )

    idx = 0
    step = quant
    while remainder > Decimal("0"):
        ticker = fractional[idx % len(fractional)]
        rounded[ticker] = (rounded[ticker] + step).quantize(quant)
        remainder = (remainder - step).quantize(quant)
        idx += 1

    return {ticker: float(value) for ticker, value in rounded.items()}


def format_portfolio_allocation(allocation: dict[str, float]) -> str:
    """Format allocation output for display."""
    if not allocation:
        return "Recommended Portfolio: (none)"

    parts = [f"{ticker} ${amount:,.2f}" for ticker, amount in allocation.items()]
    return "Recommended Portfolio: " + ", ".join(parts)
