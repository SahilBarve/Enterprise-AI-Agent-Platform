"""Structured JSON logging with structlog and automatic trace context enrichment.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Structured JSON Logging and Context Variables?
--------------------------------------------------------------------------------
1. Why not standard `print()` or plain text logs?
   In a production cluster running dozens of microservice replicas processing thousands
   of concurrent agent requests, plain text logs get interleaved and scrambled.
   Searching "which document caused this 500 error?" in CloudWatch or Datadog requires
   painful regexes.
   With structured JSON logging, every log record is a machine-readable JSON dictionary:
   {
       "event": "Document ingested and indexed",
       "level": "info",
       "timestamp": "2026-10-06T22:00:00Z",
       "tenant_id": "acme-corp",
       "correlation_id": "b3f94a28-...",
       "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
       "chunks": 12
   }
   You can instantly query: `level:error AND tenant_id:acme-corp`.

2. The Power of Python `contextvars`:
   Passing `correlation_id` and `tenant_id` as explicit arguments into every single
   function, method, and parser across 50 files would clutter code terribly.
   Instead, Python's `contextvars.ContextVar` provides task-local storage. When the
   Gateway middleware receives an HTTP request, it sets `correlation_id_ctx`.
   Whenever ANY downstream function logs an event, `add_contextvars` processor
   automatically grabs the ID from the active context!

3. Log-to-Trace Correlation:
   `add_opentelemetry_spans` inspects `opentelemetry.trace.get_current_span()`.
   If an active span exists, it injects `trace_id` and `span_id`. This allows
   observability dashboards (like Grafana or Jaeger) to jump from a log line directly
   to the exact distributed trace waterfall graph.
================================================================================
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, cast

import structlog
from opentelemetry import trace

# Context variables for request-scoped correlation, multi-tenant scope, and agent run ID
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)
tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)
run_id_ctx: ContextVar[str | None] = ContextVar("run_id", default=None)


def add_contextvars(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject contextual IDs from task-local contextvars into every log record."""
    corr_id = correlation_id_ctx.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id

    tenant_id = tenant_id_ctx.get()
    if tenant_id:
        event_dict["tenant_id"] = tenant_id

    run_id = run_id_ctx.get()
    if run_id:
        event_dict["run_id"] = run_id

    return event_dict


def add_opentelemetry_spans(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject current OpenTelemetry trace_id and span_id if an active span exists."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        span_ctx = current_span.get_span_context()
        if span_ctx.is_valid:
            event_dict["trace_id"] = format(span_ctx.trace_id, "032x")
            event_dict["span_id"] = format(span_ctx.span_id, "016x")
    return event_dict


def configure_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structlog and standard library logging for production execution.

    Args:
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: True for machine-readable JSON (production/k8s),
                     False for colorized human-readable console output (local dev).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Base processors common to both formatted output and JSON output
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_contextvars,
        add_opentelemetry_spans,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence overly verbose external loggers
    for verbose_logger in ("uvicorn.access", "urllib3", "asyncio"):
        logging.getLogger(verbose_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger instance."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
