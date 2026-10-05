"""Common platform utilities, configuration, errors, logging, and telemetry."""

from libs.common.config import PlatformSettings, get_settings
from libs.common.errors import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    GovernanceError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from libs.common.health import HealthCheckRegistry, HealthStatus
from libs.common.logging import configure_logging, get_logger
from libs.common.telemetry import (
    CorrelationIdMiddleware,
    extract_trace_context,
    inject_trace_context,
    setup_telemetry,
)

__all__ = [
    "PlatformSettings",
    "get_settings",
    "AppError",
    "NotFoundError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "GovernanceError",
    "RateLimitError",
    "configure_logging",
    "get_logger",
    "setup_telemetry",
    "inject_trace_context",
    "extract_trace_context",
    "CorrelationIdMiddleware",
    "HealthCheckRegistry",
    "HealthStatus",
]
