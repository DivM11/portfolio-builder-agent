"""Unit tests for portfolio evaluator agent."""

from src.agents.evaluator import PortfolioEvaluatorAgent


class DummyLLMService:
    def __init__(self, content: str):
        self.content = content

    def complete(self, **_kwargs):
        return {"choices": [{"message": {"content": self.content}}]}, 200

    @staticmethod
    def extract_message_text(response):
        return response["choices"][0]["message"]["content"]


def _config():
    return {
        "openrouter": {
            "default_models": {
                "analysis": "anthropic/claude-3.5-haiku",
                "evaluator": "anthropic/claude-3.5-haiku",
            },
            "outputs": {"analysis_max_tokens": 200, "evaluator_max_tokens": 300},
            "temperatures": {"analysis": 0.2, "evaluator": 0.2},
            "prompts": {
                "evaluator_system": "sys",
                "evaluator_template": "{user_input} {summary}",
                "evaluator_followup_system": "sys",
                "evaluator_followup_template": "{user_input} {summary} {previous_analysis} {applied_changes}",
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


def test_evaluator_parses_suggestions():
    content = "Portfolio is fine. {\"changes\": {\"add\": [\"NVDA\"], \"remove\": [\"TSLA\"], \"reweight\": {\"AAPL\": 0.3}}}"
    agent = PortfolioEvaluatorAgent(DummyLLMService(content), _config())

    result = agent.run_initial(
        {
            "user_input": "growth",
            "portfolio_size": 1000.0,
            "tickers": ["AAPL"],
            "weights": {"AAPL": 1.0},
            "allocation": {"AAPL": 1000.0},
            "summary_text": "summary",
        }
    )

    assert result.suggestions["add"] == ["NVDA"]
    assert result.suggestions["remove"] == ["TSLA"]
    assert result.suggestions["reweight"]["AAPL"] == 0.3


def test_evaluator_followup_returns_empty_suggestions_when_no_json():
    content = "Updated portfolio looks balanced with manageable risk."
    agent = PortfolioEvaluatorAgent(DummyLLMService(content), _config())

    result = agent.run_followup(
        {
            "user_input": "growth",
            "portfolio_size": 1000.0,
            "tickers": ["AAPL"],
            "weights": {"AAPL": 1.0},
            "allocation": {"AAPL": 1000.0},
            "summary_text": "summary",
            "previous_analysis": "prior",
        },
        {"add": ["NVDA"]},
    )

    assert result.analysis_text == content
    assert result.suggestions == {}


def test_evaluator_fail_fast_raises_on_empty_analysis_output():
    cfg = _config()
    cfg["validations"]["fail_fast"] = True
    agent = PortfolioEvaluatorAgent(DummyLLMService("   "), cfg)

    try:
        agent.run_initial(
            {
                "user_input": "growth",
                "portfolio_size": 1000.0,
                "tickers": ["AAPL"],
                "weights": {"AAPL": 1.0},
                "allocation": {"AAPL": 1000.0},
                "summary_text": "summary",
            }
        )
        assert False, "Expected validation failure"
    except ValueError as exc:
        assert "analysis.output" in str(exc)


def test_evaluator_handles_none_content_from_reasoning_model():
    """When a reasoning model returns content=None, the evaluator should
    return empty analysis_text and empty suggestions without crashing."""

    class NullContentLLMService:
        def complete(self, **_kwargs):
            return {"choices": [{"message": {"content": None}}]}, 200

        @staticmethod
        def extract_message_text(response):
            content = response["choices"][0]["message"]["content"]
            return content or ""

    cfg = _config()
    cfg["validations"]["fail_fast"] = False
    agent = PortfolioEvaluatorAgent(NullContentLLMService(), cfg)

    result = agent.run_initial(
        {
            "user_input": "growth",
            "portfolio_size": 1000.0,
            "tickers": ["AAPL"],
            "weights": {"AAPL": 1.0},
            "allocation": {"AAPL": 1000.0},
            "summary_text": "summary",
        }
    )

    assert result.analysis_text == ""
    assert result.suggestions == {}
