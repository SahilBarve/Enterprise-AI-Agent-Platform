"""Standardized health check, readiness probe, and Prometheus metrics registry."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

# Core RED metrics for all services
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


HealthCheckFn = Callable[[], Awaitable[bool]]


class HealthCheckRegistry:
    """Registry for service dependencies and readiness checks."""

    def __init__(self, service_name: str, version: str = "0.1.0") -> None:
        self.service_name = service_name
        self.version = version
        self._checks: dict[str, HealthCheckFn] = {}

    def register(self, name: str, check_fn: HealthCheckFn) -> None:
        """Register an async dependency check function."""
        self._checks[name] = check_fn

    async def liveness(self) -> JSONResponse:
        """Liveness probe: verifies process is running and event loop is responsive."""
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "service": self.service_name,
                "version": self.version,
                "timestamp": time.time(),
            },
        )

    async def readiness(self) -> JSONResponse:
        """Readiness probe: validates all registered dependencies are accessible."""
        if not self._checks:
            return JSONResponse(
                status_code=200,
                content={
                    "status": HealthStatus.HEALTHY,
                    "service": self.service_name,
                    "dependencies": {},
                    "timestamp": time.time(),
                },
            )

        results: dict[str, dict[str, Any]] = {}
        all_healthy = True

        async def run_check(name: str, fn: HealthCheckFn) -> tuple[str, bool, str | None]:
            try:
                healthy = await asyncio.wait_for(fn(), timeout=3.0)
                return name, healthy, None if healthy else "Check returned false"
            except Exception as exc:
                return name, False, str(exc)

        tasks = [run_check(name, fn) for name, fn in self._checks.items()]
        completed = await asyncio.gather(*tasks)

        for name, healthy, error_msg in completed:
            results[name] = {
                "status": HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
            }
            if error_msg:
                results[name]["error"] = error_msg
            if not healthy:
                all_healthy = False

        status_code = 200 if all_healthy else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY,
                "service": self.service_name,
                "dependencies": results,
                "timestamp": time.time(),
            },
        )

    @staticmethod
    def metrics() -> Response:
        """Generate Prometheus exposition format metrics."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
