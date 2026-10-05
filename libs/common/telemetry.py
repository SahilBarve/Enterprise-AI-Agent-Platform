"""OpenTelemetry distributed tracing setup and correlation ID propagation utilities."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from libs.common.logging import correlation_id_ctx, get_logger

logger = get_logger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


def setup_telemetry(
    service_name: str,
    otlp_endpoint: str | None = None,
    insecure: bool = True,
    use_batch_exporter: bool = True,
) -> TracerProvider:
    """Initialize OpenTelemetry tracer provider with OTLP gRPC exporter."""
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if (
        otlp_endpoint
        and otlp_endpoint.strip()
        and otlp_endpoint.lower() not in ("none", "false", "0")
    ):
        try:
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=insecure,
            )
            processor = (
                BatchSpanProcessor(exporter)
                if use_batch_exporter
                else SimpleSpanProcessor(exporter)
            )
            provider.add_span_processor(processor)
            logger.info("OTel exporter configured", endpoint=otlp_endpoint, service=service_name)
        except Exception as exc:
            logger.warning(
                "Failed to configure OTLP exporter, running without remote export", error=str(exc)
            )

    trace.set_tracer_provider(provider)
    return provider


def inject_trace_context(carrier: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inject W3C trace context (traceparent) into a carrier dict for RabbitMQ/MCP."""
    if carrier is None:
        carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_trace_context(carrier: dict[str, Any]) -> Any:
    """Extract W3C trace context from incoming carrier dict (HTTP/RabbitMQ/MCP)."""
    return TraceContextTextMapPropagator().extract(carrier)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware ensuring every request has a correlation ID."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Check incoming headers for existing correlation ID
        incoming_id = request.headers.get(CORRELATION_ID_HEADER) or request.headers.get(
            REQUEST_ID_HEADER
        )
        corr_id = incoming_id.strip() if incoming_id else str(uuid.uuid4())

        # Set in contextvar for structlog
        token = correlation_id_ctx.set(corr_id)

        # Set correlation ID attribute on active span if present
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("app.correlation_id", corr_id)

        try:
            response = await call_next(request)
            response.headers[CORRELATION_ID_HEADER] = corr_id
            return response
        finally:
            correlation_id_ctx.reset(token)
