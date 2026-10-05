"""Unit tests for the API Gateway service endpoints, middleware, and error handling."""

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from libs.common.config import PlatformSettings
from libs.common.errors import NotFoundError
from libs.common.telemetry import CORRELATION_ID_HEADER
from services.gateway.main import create_gateway_app


class SampleRequest(BaseModel):
    name: str
    age: int


@pytest.fixture
def test_client() -> TestClient:
    """Create a TestClient with custom test settings."""
    settings = PlatformSettings(
        ENVIRONMENT="testing",
        GATEWAY_PORT=8000,
        OTEL_EXPORTER_OTLP_ENDPOINT=None,
    )
    app = create_gateway_app(settings=settings)

    # Add test route that raises AppError
    @app.get("/test-not-found")
    def sample_not_found() -> None:
        raise NotFoundError("Resource not found", resource_type="user", resource_id="usr-123")

    # Add test route that requires schema validation
    @app.post("/test-schema")
    def sample_schema(body: SampleRequest) -> dict[str, str]:
        return {"status": "ok", "name": body.name}

    return TestClient(app)


@pytest.mark.unit
def test_gateway_healthz(test_client: TestClient) -> None:
    """Validate /healthz returns 200 OK with correlation header."""
    response = test_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "api-gateway"
    assert CORRELATION_ID_HEADER in response.headers


@pytest.mark.unit
def test_gateway_readyz(test_client: TestClient) -> None:
    """Validate /readyz returns 200 OK."""
    response = test_client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.unit
def test_gateway_metrics(test_client: TestClient) -> None:
    """Validate /metrics returns Prometheus text format."""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.unit
def test_gateway_api_status(test_client: TestClient) -> None:
    """Validate /api/v1/status endpoint."""
    response = test_client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["environment"] == "testing"


@pytest.mark.unit
def test_gateway_rfc7807_app_error_handling(test_client: TestClient) -> None:
    """Validate that unhandled AppErrors are converted to RFC 7807 problem details."""
    response = test_client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["type"] == "urn:aiops:error:resource-not-found"
    assert data["title"] == "RESOURCE_NOT_FOUND"
    assert data["status"] == 404
    assert data["detail"] == "Resource not found"
    assert data["invalid_params"]["resource_type"] == "user"
    assert data["invalid_params"]["resource_id"] == "usr-123"
    assert data["instance"] == "/test-not-found"
    assert CORRELATION_ID_HEADER in response.headers


@pytest.mark.unit
def test_gateway_rfc7807_validation_error_handling(test_client: TestClient) -> None:
    """Validate schema validation errors return formatted RFC 7807 details."""
    # Send invalid body (missing age, name not string)
    response = test_client.post("/test-schema", json={"name": 123})
    assert response.status_code == 422
    data = response.json()
    assert data["type"] == "urn:aiops:error:validation-error"
    assert data["title"] == "VALIDATION_ERROR"
    assert data["status"] == 422
    assert "invalid_params" in data
    assert data["instance"] == "/test-schema"
