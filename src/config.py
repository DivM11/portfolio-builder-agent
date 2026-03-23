"""Configuration loading utilities."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(base_path: Path | None = None) -> dict[str, Any]:
    """Load config.yml and merge in environment-variable secrets.

    API keys are read from environment variables (e.g. ``OPENROUTER_API_KEY``,
    ``MASSIVE_API_KEY``).  Pass them via ``docker run -e`` or export them in
    your shell — no ``.env`` file is required.
    """
    root = base_path or Path(__file__).resolve().parents[1]
    config_path = root / "config.yml"
    logger.info("Loading configuration from %s", config_path)
    config = OmegaConf.load(config_path)

    # --- OpenRouter API key ---
    key_env_var = "OPENROUTER_API_KEY"
    with suppress(AttributeError, KeyError, TypeError):
        key_env_var = config.openrouter.api.key_env_var

    api_key = os.getenv(key_env_var)
    if api_key:
        config.openrouter.api.api_key = api_key

    # --- Massive.com (formerly Polygon.io) API key ---
    massive_key_env_var = "MASSIVE_API_KEY"
    with suppress(AttributeError, KeyError, TypeError):
        massive_key_env_var = config.massive.api.key_env_var

    massive_api_key = os.getenv(massive_key_env_var)
    if massive_api_key:
        try:
            config.massive.api.api_key = massive_api_key
        except (AttributeError, KeyError, TypeError):
            if not hasattr(config, "massive"):
                config.massive = DictConfig({"api": {"api_key": massive_api_key, "key_env_var": massive_key_env_var}})
            elif not hasattr(config.massive, "api"):
                config.massive.api = DictConfig({"api_key": massive_api_key, "key_env_var": massive_key_env_var})
            else:
                config.massive.api.api_key = massive_api_key

    OmegaConf.resolve(config)
    container = OmegaConf.to_container(config, resolve=True)
    return cast(dict[str, Any], container)
