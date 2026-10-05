"""Unit tests for structured logging and context variable injection."""

import pytest

from libs.common.logging import (
    add_contextvars,
    configure_logging,
    correlation_id_ctx,
    get_logger,
    run_id_ctx,
    tenant_id_ctx,
)


@pytest.mark.unit
def test_add_contextvars_injects_metadata() -> None:
    """Validate that active contextvars are automatically injected into log dict."""
    corr_token = correlation_id_ctx.set("corr-test-123")
    tenant_token = tenant_id_ctx.set("tenant-acme")
    run_token = run_id_ctx.set("run-456")

    try:
        event_dict: dict = {"event": "user_action", "status": "ok"}
        processed = add_contextvars(None, "info", event_dict)

        assert processed["correlation_id"] == "corr-test-123"
        assert processed["tenant_id"] == "tenant-acme"
        assert processed["run_id"] == "run-456"
        assert processed["event"] == "user_action"
    finally:
        correlation_id_ctx.reset(corr_token)
        tenant_id_ctx.reset(tenant_token)
        run_id_ctx.reset(run_token)


@pytest.mark.unit
def test_add_contextvars_when_empty() -> None:
    """Validate processor leaves dictionary unmodified when contextvars are unset."""
    event_dict: dict = {"event": "system_boot"}
    processed = add_contextvars(None, "info", event_dict)
    assert "correlation_id" not in processed
    assert "tenant_id" not in processed
    assert "run_id" not in processed
    assert processed["event"] == "system_boot"


@pytest.mark.unit
def test_configure_logging_executes() -> None:
    """Validate configure_logging runs and produces a functional logger."""
    configure_logging(log_level="DEBUG", json_format=True)
    logger = get_logger("test-service")
    assert logger is not None
