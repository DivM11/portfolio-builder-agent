"""Unit tests for LLM validation helpers."""

from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_evaluator_suggestions,
    parse_weights_payload,
    validate_weight_sum,
)


def test_extract_valid_tickers_filters_noise():
    tickers = extract_valid_tickers("AAPL, ???, MSFT, AAPL", delimiter=",")

    assert tickers == ["AAPL", "MSFT"]


def test_extract_valid_tickers_space_separated():
    tickers = extract_valid_tickers("AAPL GOOGLE\nMSFT", delimiter=",")

    assert tickers == ["AAPL", "GOOGLE", "MSFT"]


def test_has_valid_tickers():
    assert has_valid_tickers(["AAPL", "MSFT"]) is True
    assert has_valid_tickers(["", "???"]) is False


def test_parse_weights_payload_object_and_nested():
    nested = parse_weights_payload('{"weights": {"AAPL": 0.6, "MSFT": 0.4}}')
    plain = parse_weights_payload('{"AAPL": 0.7, "MSFT": 0.3}')

    assert nested["AAPL"] == 0.6
    assert plain["MSFT"] == 0.3


def test_validate_weight_sum():
    ok, total_ok = validate_weight_sum({"AAPL": 0.6, "MSFT": 0.4})
    bad, total_bad = validate_weight_sum({"AAPL": 0.6, "MSFT": 0.2})

    assert ok is True
    assert round(total_ok, 2) == 1.0
    assert bad is False
    assert round(total_bad, 2) == 0.8


def test_parse_weights_payload_markdown_fenced():
    text = '```json\n{"weights": {"AAPL": 0.6, "MSFT": 0.4}}\n```'
    result = parse_weights_payload(text)

    assert result == {"AAPL": 0.6, "MSFT": 0.4}


def test_parse_weights_payload_with_think_tags():
    text = '<think>\nLet me analyze...\n</think>\n{"weights": {"NVDA": 0.5, "AMD": 0.5}}'
    result = parse_weights_payload(text)

    assert result == {"NVDA": 0.5, "AMD": 0.5}


def test_parse_weights_payload_with_surrounding_text():
    text = 'Here are the weights:\n{"weights": {"GOOG": 0.7, "META": 0.3}}\nDone.'
    result = parse_weights_payload(text)

    assert result == {"GOOG": 0.7, "META": 0.3}


def test_parse_evaluator_suggestions():
    result = parse_evaluator_suggestions(
        '{"changes": {"add": ["nvda"], "remove": ["tsla"], "reweight": {"aapl": 0.4}}}'
    )

    assert result["add"] == ["NVDA"]
    assert result["remove"] == ["TSLA"]
    assert result["reweight"]["AAPL"] == 0.4


def test_parse_evaluator_suggestions_empty_string():
    """Empty string (e.g. from reasoning model null content) returns empty dict."""
    assert parse_evaluator_suggestions("") == {}


def test_parse_evaluator_suggestions_none_input():
    """None input must not raise."""
    assert parse_evaluator_suggestions(None) == {}


def test_parse_weights_payload_empty_string():
    assert parse_weights_payload("") == {}


def test_parse_weights_payload_none_input():
    assert parse_weights_payload(None) == {}
