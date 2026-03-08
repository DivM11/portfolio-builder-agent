"""Unit tests for prompt input/output validators."""

import pytest

from src.prompt_validation import (
    AnalysisPromptValidator,
    PortfolioPromptValidator,
    PromptValidationError,
    PromptValidationRunner,
    TickerPromptValidator,
)


def test_ticker_validator_input_and_output_errors():
    validator = TickerPromptValidator()

    assert validator.validate_input({"user_query": "", "max_tickers": 0})
    assert validator.validate_output({"raw_output": "none", "parsed_output": []})


def test_portfolio_validator_output_checks_weight_sum():
    validator = PortfolioPromptValidator()

    errors = validator.validate_output(
        {
            "raw_output": '{"weights": {"AAPL": 0.2, "MSFT": 0.2}}',
            "parsed_output": {"AAPL": 0.2, "MSFT": 0.2},
        }
    )
    assert any("sum" in err.lower() for err in errors)


def test_analysis_validator_requires_non_empty_output():
    validator = AnalysisPromptValidator()

    errors = validator.validate_output({"raw_output": "   ", "parsed_output": {}})
    assert errors


def test_validation_runner_respects_disabled_toggle():
    runner = PromptValidationRunner({"enabled": False, "fail_fast": True})
    errors = runner.validate_input("ticker", TickerPromptValidator(), {"user_query": "", "max_tickers": 0})

    assert errors == []


def test_validation_runner_fail_fast_raises():
    runner = PromptValidationRunner({"enabled": True, "validate_input": True, "fail_fast": True})

    with pytest.raises(PromptValidationError):
        runner.validate_input("ticker", TickerPromptValidator(), {"user_query": "", "max_tickers": 0})
