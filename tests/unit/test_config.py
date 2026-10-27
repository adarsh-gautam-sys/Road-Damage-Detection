"""Unit tests for application configuration loading and environment overrides (`U09`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdai.config import ConfigError, load_app_config


def test_u09_env_vars_override_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """U09: Environment variables override settings from `configs/app.yaml`."""
    monkeypatch.setenv("RDAI_DATABASE_URL", "sqlite:///./custom_override.db")
    monkeypatch.setenv("RDAI_MEDIA_DIR", "custom_media")
    monkeypatch.setenv("RDAI_DEFAULT_CONF", "0.42")
    monkeypatch.setenv("RDAI_MAX_PHOTO_MB", "25")
    monkeypatch.setenv("RDAI_CORS_ORIGINS", "https://road.example.org,https://admin.example.org")

    cfg = load_app_config()
    assert cfg.database_url == "sqlite:///./custom_override.db"
    assert cfg.media_dir == "custom_media"
    assert cfg.default_conf == pytest.approx(0.42)
    assert cfg.max_photo_mb == 25
    assert cfg.cors_origins == ["https://road.example.org", "https://admin.example.org"]


def test_u09_broken_yaml_fails_with_readable_message(tmp_path: Path) -> None:
    """U09: Broken or malformed YAML raises `ConfigError` with a readable message."""
    bad_yaml = tmp_path / "broken_app.yaml"
    bad_yaml.write_text("app:\n  title: [unclosed list\n  invalid: : :", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_app_config(app_yaml_path=bad_yaml)

    msg = str(exc_info.value)
    assert "broken_app.yaml" in msg
    assert "Invalid YAML" in msg or "Failed to parse" in msg


def test_u09_missing_yaml_fails_with_readable_message(tmp_path: Path) -> None:
    """U09: Missing configuration file raises `ConfigError` naming the missing path."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError) as exc_info:
        load_app_config(app_yaml_path=missing)
    assert "does_not_exist.yaml" in str(exc_info.value)
