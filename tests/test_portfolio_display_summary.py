"""Unit tests for portfolio display summary formatting."""

import pytest

from src.portfolio_display_summary import PortfolioDisplaySummary


def test_format_suggestions_human_readable():
    summary = PortfolioDisplaySummary()
    text = summary.format_suggestions(
        {
            "add": ["FSLR", "CRWD"],
            "remove": ["TSLA"],
            "reweight": {"MSFT": 0.15, "NVDA": 0.3},
        }
    )

    assert "Suggested portfolio changes:" in text
    assert "Add: FSLR, CRWD" in text
    assert "Remove: TSLA" in text
    assert "MSFT: 15.00%" in text


@pytest.mark.parametrize("payload", [{"add": [], "remove": [], "reweight": {}}, {}])
def test_format_suggestions_no_changes(payload):
    summary = PortfolioDisplaySummary()
    text = summary.format_suggestions(payload)
    assert text == "No suggested changes."


def test_format_suggestions_shows_reweight_when_present():
    summary = PortfolioDisplaySummary()
    text = summary.format_suggestions({"add": [], "remove": [], "reweight": {"AAPL": 0.6, "MSFT": 0.4}})
    assert "Reweight:" in text
    assert "AAPL: 60.00%" in text


def test_format_suggestions_shows_add_even_without_reweight():
    summary = PortfolioDisplaySummary()
    text = summary.format_suggestions({"add": ["GOOG"], "remove": [], "reweight": {}})
    assert "Add: GOOG" in text
    assert "Reweight: None" in text


def test_format_portfolio_header_empty():
    summary = PortfolioDisplaySummary()
    assert summary.format_portfolio_header([]) == "Recommended Portfolio Tickers: (none)"
