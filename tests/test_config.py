"""Unit tests for config loading."""

from pathlib import Path

from src.config import load_config


def _write_config(root: Path) -> None:
    config_text = (
        "app:\n"
        '  layout: "wide"\n'
        "massive:\n"
        "  api:\n"
        '    key_env_var: "MASSIVE_API_KEY"\n'
        "openrouter:\n"
        "  api:\n"
        '    key_env_var: "OPENROUTER_API_KEY"\n'
        '    base_url: "https://openrouter.ai/api/v1"\n'
    )
    (root / "config.yml").write_text(config_text, encoding="utf-8")


def test_load_config_without_secrets(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    config = load_config(base_path=tmp_path)

    assert config["app"]["layout"] == "wide"
    assert "api_key" not in config.get("openrouter", {}).get("api", {})


def test_load_config_with_env_vars(monkeypatch, tmp_path: Path) -> None:
    """API keys injected via environment variables are picked up."""
    _write_config(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive-key")

    config = load_config(base_path=tmp_path)

    assert config["openrouter"]["api"]["api_key"] == "test-key"
    assert config["massive"]["api"]["api_key"] == "test-massive-key"


def test_load_config_uses_custom_env_var_names(monkeypatch, tmp_path: Path) -> None:
    config_text = 'openrouter:\n  api:\n    key_env_var: "OR_KEY"\nmassive:\n  api:\n    key_env_var: "MASSIVE_KEY"\n'
    (tmp_path / "config.yml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("OR_KEY", "or-value")
    monkeypatch.setenv("MASSIVE_KEY", "massive-value")

    config = load_config(base_path=tmp_path)

    assert config["openrouter"]["api"]["api_key"] == "or-value"
    assert config["massive"]["api"]["api_key"] == "massive-value"


def test_load_config_creates_missing_massive_sections(monkeypatch, tmp_path: Path) -> None:
    config_text = "openrouter:\n  api:\n    key_env_var: OPENROUTER_API_KEY\n"
    (tmp_path / "config.yml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-value")

    config = load_config(base_path=tmp_path)

    assert config["massive"]["api"]["api_key"] == "massive-value"
    assert config["massive"]["api"]["key_env_var"] == "MASSIVE_API_KEY"
