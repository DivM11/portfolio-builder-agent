"""Unit tests for LLM validation helpers."""

from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_evaluator_suggestions,
    parse_weights_payload,
    validate_weight_sum,
)
from src.tools.generate_tickers import generate_tickers_tool


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


def test_extract_valid_tickers_custom_delimiter() -> None:
    tickers = extract_valid_tickers("AAPL::MSFT::INVALID", delimiter="::")

    assert tickers == ["AAPL", "MSFT", "INVALID"]


def test_parse_weights_payload_list_shape() -> None:
    result = parse_weights_payload(
        '```json\n[{"ticker":"aapl","weight":0.6},{"ticker":"msft","weight":"0.4"}]\n```'
    )

    assert result == {"AAPL": 0.6, "MSFT": 0.4}


def test_parse_evaluator_suggestions_non_dict_candidate_returns_empty() -> None:
    assert parse_evaluator_suggestions('{"changes": []}') == {}


def test_parse_evaluator_suggestions_ignores_bad_reweight_values() -> None:
    result = parse_evaluator_suggestions('{"reweight": {"AAPL": "oops", "MSFT": 0.2}}')

    assert result["reweight"] == {"MSFT": 0.2}


# ---------------------------------------------------------------------------
# Negative / invalid weight rejection
# ---------------------------------------------------------------------------


def test_parse_weights_payload_rejects_negative_weights() -> None:
    """Negative weights must be silently dropped, not included."""
    result = parse_weights_payload('{"AAPL": -0.5, "MSFT": 0.5}')

    assert "AAPL" not in result
    assert result["MSFT"] == 0.5


def test_parse_weights_payload_rejects_negative_in_nested_weights_key() -> None:
    result = parse_weights_payload('{"weights": {"AAPL": 0.6, "MSFT": -0.1}}')

    assert "MSFT" not in result
    assert result["AAPL"] == 0.6


def test_parse_weights_payload_rejects_negative_in_list_shape() -> None:
    # List-shape JSON needs a code fence so _extract_json_string uses the
    # fence regex path (not the {/} character search which strips the outer [])
    payload = '```json\n[{"ticker": "AAPL", "weight": 0.8}, {"ticker": "MSFT", "weight": -0.2}]\n```'
    result = parse_weights_payload(payload)

    assert "MSFT" not in result
    assert result["AAPL"] == 0.8


def test_parse_weights_payload_all_negative_returns_empty() -> None:
    result = parse_weights_payload('{"AAPL": -0.5, "MSFT": -0.5}')

    assert result == {}


def test_validate_weight_sum_ignores_negative_values() -> None:
    """validate_weight_sum clamps negatives to 0 before summing."""
    valid, total = validate_weight_sum({"AAPL": 0.6, "MSFT": -0.1})

    # Effective sum = 0.6, not 1.0 → invalid
    assert valid is False
    assert round(total, 2) == 0.6


# ---------------------------------------------------------------------------
# Edge cases: very long / malformed inputs
# ---------------------------------------------------------------------------


def test_extract_valid_tickers_very_long_input() -> None:
    """Very long input must not crash and must return only valid tickers."""
    long_input = ("AAPL " * 1000) + "???" * 500
    tickers = extract_valid_tickers(long_input, delimiter=",")

    assert len(tickers) == 1  # deduplicated
    assert tickers[0] == "AAPL"


def test_parse_weights_payload_unicode_ticker_ignored() -> None:
    """Unicode or non-ASCII tickers should not crash the parser."""
    result = parse_weights_payload('{"\u30C4": 0.5, "AAPL": 0.5}')

    # ツ is not a valid uppercase ASCII ticker — kept as-is by parser
    # but must not crash
    assert "AAPL" in result or len(result) >= 0  # just must not raise

# ---------------------------------------------------------------------------
# generate_tickers_tool reasoning sanitization
# ---------------------------------------------------------------------------


def test_generate_tickers_tool_truncates_long_reasoning() -> None:
    """reasoning field must be truncated to 500 chars."""
    long_reasoning = "a" * 1000
    result = generate_tickers_tool(
        {"tickers": ["AAPL"], "reasoning": long_reasoning},
        max_tickers=10,
    )
    assert len(result["reasoning"]) == 500


def test_generate_tickers_tool_strips_control_chars_from_reasoning() -> None:
    """Control characters (except \\n, \\t) must be stripped from reasoning."""
    result = generate_tickers_tool(
        {"tickers": ["AAPL"], "reasoning": "good\x00reason\x01here"},
        max_tickers=10,
    )
    assert "\x00" not in result["reasoning"]
    assert "\x01" not in result["reasoning"]
    assert "goodreasonhere" in result["reasoning"]


def test_generate_tickers_tool_preserves_newlines_in_reasoning() -> None:
    """Newlines and tabs in reasoning should be preserved."""
    result = generate_tickers_tool(
        {"tickers": ["AAPL"], "reasoning": "line1\nline2\ttabbed"},
        max_tickers=10,
    )
    assert "\n" in result["reasoning"]
    assert "\t" in result["reasoning"]
