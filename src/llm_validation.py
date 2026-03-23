"""Validation helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def extract_valid_tickers(text: str, delimiter: str) -> list[str]:
    split_pattern = r"[\s,;|]+"
    if delimiter and delimiter not in [",", ";", "|"]:
        split_pattern = rf"(?:\s+|{re.escape(delimiter)}|,|;|\|)+"

    candidates = [item.strip().upper() for item in re.split(split_pattern, text) if item.strip()]
    deduped: list[str] = []
    for ticker in candidates:
        if ticker not in deduped and TICKER_PATTERN.match(ticker):
            deduped.append(ticker)
    return deduped


def has_valid_tickers(tickers: Iterable[str]) -> bool:
    return any(TICKER_PATTERN.match(str(ticker).upper()) for ticker in tickers)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json_string(text: str) -> str:
    """Extract a JSON object from LLM output that may contain markdown
    code fences, ``<think>`` blocks, or surrounding prose."""
    if not text:
        return ""
    cleaned = _THINK_TAG_RE.sub("", text).strip()

    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        return fence.group(1).strip()

    for open_ch, close_ch in ["{", "}"], ["[", "]"]:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            return cleaned[start : end + 1]

    return cleaned


def parse_weights_payload(text: str) -> dict[str, float]:
    cleaned = _extract_json_string(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    weights: dict[str, float] = {}
    if isinstance(payload, dict):
        source = payload.get("weights", payload)
        if isinstance(source, dict):
            for ticker, value in source.items():
                try:
                    weights[str(ticker).upper()] = float(value)
                except (TypeError, ValueError):
                    continue
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "ticker" in item and "weight" in item:
                try:
                    weights[str(item["ticker"]).upper()] = float(item["weight"])
                except (TypeError, ValueError):
                    continue

    return weights


def validate_weight_sum(weights: dict[str, float], tolerance: float = 0.02) -> tuple[bool, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    return abs(total - 1.0) <= tolerance, total


def parse_evaluator_suggestions(text: str) -> dict[str, object]:
    cleaned = _extract_json_string(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    candidate = payload.get("changes", payload)
    if not isinstance(candidate, dict):
        return {}

    add_raw = candidate.get("add", [])
    remove_raw = candidate.get("remove", [])
    reweight_raw = candidate.get("reweight", {})

    add = [str(item).upper() for item in add_raw] if isinstance(add_raw, list) else []
    remove = [str(item).upper() for item in remove_raw] if isinstance(remove_raw, list) else []
    reweight: dict[str, float] = {}
    if isinstance(reweight_raw, dict):
        for ticker, value in reweight_raw.items():
            try:
                reweight[str(ticker).upper()] = float(value)
            except (TypeError, ValueError):
                continue

    return {
        "add": add,
        "remove": remove,
        "reweight": reweight,
    }
