"""Unit tests for platform configuration and settings."""

import pytest

from libs.common.config import PlatformSettings, get_settings


@pytest.mark.unit
def test_platform_settings_defaults() -> None:
    """Validate default configuration values."""
    settings = PlatformSettings()
    assert settings.ENVIRONMENT == "development"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.GATEWAY_PORT == 8000
    assert settings.is_production is False
    assert settings.is_testing is False
    assert "localhost" in settings.DATABASE_URL
    assert "localhost" in settings.REDIS_URL
    assert "localhost" in settings.RABBITMQ_URL


@pytest.mark.unit
def test_platform_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate settings override via environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GATEWAY_PORT", "9000")
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-4o-custom")

    settings = PlatformSettings()
    assert settings.ENVIRONMENT == "production"
    assert settings.is_production is True
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.GATEWAY_PORT == 9000
    assert settings.OPENAI_MODEL_PRIMARY == "gpt-4o-custom"


@pytest.mark.unit
def test_get_settings_cached() -> None:
    """Validate get_settings provides a cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
