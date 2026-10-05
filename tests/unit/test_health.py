"""Unit tests for health check registry, readiness probes, and Prometheus metrics."""

import json

import pytest

from libs.common.health import HealthCheckRegistry, HealthStatus


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_registry_liveness() -> None:
    """Validate liveness probe returns 200 OK with service metadata."""
    registry = HealthCheckRegistry(service_name="test-service", version="1.0.0")
    response = await registry.liveness()

    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert data["status"] == "ok"
    assert data["service"] == "test-service"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_registry_readiness_all_healthy() -> None:
    """Validate readiness probe returns 200 when all dependencies pass."""
    registry = HealthCheckRegistry(service_name="test-service")

    async def check_db() -> bool:
        return True

    async def check_redis() -> bool:
        return True

    registry.register("postgres", check_db)
    registry.register("redis", check_redis)

    response = await registry.readiness()
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert data["status"] == HealthStatus.HEALTHY
    assert data["dependencies"]["postgres"]["status"] == HealthStatus.HEALTHY
    assert data["dependencies"]["redis"]["status"] == HealthStatus.HEALTHY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_registry_readiness_with_failure() -> None:
    """Validate readiness probe returns 503 when any dependency fails."""
    registry = HealthCheckRegistry(service_name="test-service")

    async def check_db() -> bool:
        return True

    async def check_qdrant() -> bool:
        raise ConnectionError("Qdrant unreachable on port 6333")

    registry.register("postgres", check_db)
    registry.register("qdrant", check_qdrant)

    response = await registry.readiness()
    assert response.status_code == 503
    data = json.loads(response.body.decode())
    assert data["status"] == HealthStatus.UNHEALTHY
    assert data["dependencies"]["postgres"]["status"] == HealthStatus.HEALTHY
    assert data["dependencies"]["qdrant"]["status"] == HealthStatus.UNHEALTHY
    assert "Qdrant unreachable" in data["dependencies"]["qdrant"]["error"]


@pytest.mark.unit
def test_health_registry_metrics_export() -> None:
    """Validate Prometheus metrics output generation."""
    registry = HealthCheckRegistry(service_name="test-service")
    response = registry.metrics()
    assert response.status_code == 200
    assert response.media_type.startswith("text/plain")
    assert b"http_requests_total" in response.body or b"#" in response.body
