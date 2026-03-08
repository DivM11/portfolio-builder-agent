"""Unit tests for portfolio creator agent."""

import pandas as pd

from src.agents.creator import PortfolioCreatorAgent


class DummyLLMService:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def complete(self, **_kwargs):
        return {"choices": [{"message": {"content": self.outputs.pop(0)}}]}, 200

    @staticmethod
    def extract_message_text(response):
        return response["choices"][0]["message"]["content"]


def _config():
    return {
        "dashboard": {"ticker_delimiter": ","},
        "stocks": {
            "max_tickers": 5,
            "history_period": "1y",
            "financials_period": "quarterly",
            "financials_metrics": ["Total Revenue"],
        },
        "massive": {"api": {"api_key": "k"}},
        "openrouter": {
            "models": {
                "ticker": "anthropic/claude-3.5-haiku",
                "weights": "anthropic/claude-3.5-haiku",
            },
            "outputs": {
                "ticker_max_tokens": 100,
                "weights_max_tokens": 200,
            },
            "temperatures": {"ticker": 0.2, "weights": 0.1},
            "prompts": {
                "ticker_system": "sys {num_tickers}",
                "ticker_template": "{user_input}",
                "weights_system": "sys",
                "weights_template": "{user_input} {tickers} {summary}",
            },
        },
        "validations": {
            "enabled": True,
            "validate_input": True,
            "validate_output": True,
            "fail_fast": False,
            "prompts": {"ticker": True, "portfolio": True, "analysis": True},
        },
    }


def _fetcher(**kwargs):
    ticker = kwargs["ticker"]
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    history = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
    return {"history": history, "financials": pd.DataFrame(), "history_status": "ok", "ticker": ticker}


def test_run_initial_builds_portfolio_result():
    llm = DummyLLMService([
        "AAPL, MSFT, NVDA",
        '{"weights": {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}}',
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=_config(),
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    result = agent.run_initial({"user_input": "growth with NVDA", "portfolio_size": 1000.0})

    assert set(result.tickers) == {"AAPL", "MSFT", "NVDA"}
    assert round(sum(result.weights.values()), 3) == 1.0
    assert round(sum(result.allocation.values()), 2) == 1000.0


def test_run_initial_with_legacy_config_keys():
    legacy_config = _config()
    legacy_config["dashboard"] = {}

    llm = DummyLLMService([
        "AAPL, MSFT",
        '{"weights": {"AAPL": 0.6, "MSFT": 0.4}}',
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=legacy_config,
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    result = agent.run_initial({"user_input": "tech portfolio", "portfolio_size": 1000.0})

    assert result.tickers == ["AAPL", "MSFT"]
    assert round(sum(result.weights.values()), 3) == 1.0


def test_run_followup_applies_add_remove_feedback():
    llm = DummyLLMService([
        "AAPL, TSLA",
        '{"weights": {"AAPL": 0.6, "TSLA": 0.4}}',
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=_config(),
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    result = agent.run_followup({"user_input": "refine", "portfolio_size": 1000.0}, {"add": ["NVDA"]})

    assert set(result.tickers) == {"AAPL", "TSLA"}


def test_run_initial_raises_when_no_valid_tickers():
    llm = DummyLLMService([
        "???",
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=_config(),
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    try:
        agent.run_initial({"user_input": "bad", "portfolio_size": 1000.0})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "No valid ticker symbols" in str(exc)


def test_run_initial_collects_validation_errors_when_not_fail_fast():
    cfg = _config()
    cfg["validations"]["fail_fast"] = False
    llm = DummyLLMService([
        "AAPL, MSFT",
        '{"weights": {"AAPL": 0.2, "MSFT": 0.2}}',
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=cfg,
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    result = agent.run_initial({"user_input": "balanced", "portfolio_size": 1000.0})
    errors = result.metadata.get("validation_errors", [])

    assert errors
    assert any("portfolio.output" in err for err in errors)


def test_run_initial_fail_fast_raises_on_invalid_weights_output():
    cfg = _config()
    cfg["validations"]["fail_fast"] = True
    llm = DummyLLMService([
        "AAPL, MSFT",
        '{"weights": {"AAPL": 0.2, "MSFT": 0.2}}',
    ])
    agent = PortfolioCreatorAgent(
        llm_service=llm,
        config=cfg,
        massive_client_factory=lambda _api_key: object(),
        stock_data_fetcher=_fetcher,
    )

    try:
        agent.run_initial({"user_input": "balanced", "portfolio_size": 1000.0})
        assert False, "Expected validation failure"
    except ValueError as exc:
        assert "portfolio.output" in str(exc)
