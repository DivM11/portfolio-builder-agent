"""Unit tests for structured logging configuration."""

import io
import json
import logging

from src.logging_config import configure_logging, set_log_context


def test_configure_logging_json_with_correlation_ids() -> None:
    stream = io.StringIO()
    configure_logging(
        {
            "level": "INFO",
            "format": "json",
            "include_timestamps": False,
            "include_correlation_ids": True,
        },
        stream=stream,
    )
    set_log_context(session_id="s-1", run_id="r-1")

    logger = logging.getLogger("tests.logging")
    logger.info("hello %s", "world")

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == "tests.logging"
    assert payload["message"] == "hello world"
    assert payload["session_id"] == "s-1"
    assert payload["run_id"] == "r-1"
