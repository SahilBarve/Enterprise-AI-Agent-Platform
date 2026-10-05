"""Unit tests for OpenTelemetry setup, trace context propagation, and correlation middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace

from libs.common.telemetry import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    extract_trace_context,
    inject_trace_context,
    setup_telemetry,
)


@pytest.mark.unit
def test_setup_telemetry_provider() -> None:
    """Validate setup_telemetry initializes TracerProvider."""
    provider = setup_telemetry(service_name="test-telemetry", otlp_endpoint=None)
    assert provider is not None
    current_provider = trace.get_tracer_provider()
    assert current_provider is not None


@pytest.mark.unit
def test_trace_context_injection_and_extraction() -> None:
    """Validate W3C TraceContext injection and extraction across boundaries."""
    tracer = trace.get_tracer("test-tracer")
    with tracer.start_as_current_span("parent-operation"):
        carrier: dict = {}
        inject_trace_context(carrier)

        assert "traceparent" in carrier
        traceparent = carrier["traceparent"]
        assert traceparent.startswith("00-")

        # Extract in downstream context
        extracted_ctx = extract_trace_context(carrier)
        assert extracted_ctx is not None


@pytest.mark.unit
def test_correlation_id_middleware_generates_id() -> None:
    """Validate middleware assigns a new UUID if no correlation header is provided."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    def endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert CORRELATION_ID_HEADER in response.headers
    generated_id = response.headers[CORRELATION_ID_HEADER]
    assert len(generated_id) > 10


@pytest.mark.unit
def test_correlation_id_middleware_preserves_incoming_id() -> None:
    """Validate middleware preserves incoming X-Correlation-ID header."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    def endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    custom_id = "corr-custom-abc-123"
    response = client.get("/test", headers={CORRELATION_ID_HEADER: custom_id})

    assert response.status_code == 200
    assert response.headers[CORRELATION_ID_HEADER] == custom_id
