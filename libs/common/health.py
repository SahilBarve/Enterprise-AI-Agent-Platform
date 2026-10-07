"""Standardized health check, readiness probe, and Prometheus metrics registry.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why separate Liveness, Readiness, and RED Metrics in Kubernetes?
--------------------------------------------------------------------------------
In modern cloud-native Kubernetes environments, understanding probe semantics
is vital to prevent production outages:

1. Liveness Probe (`/healthz`):
   - Question: "Is this container process alive, or is Python completely deadlocked?"
   - Kubernetes Action: If `/healthz` fails (returns non-200), Kubelet KILLS and
     RESTARTS the pod container.
   - Golden Rule: Never check external databases in `/healthz`! If PostgreSQL has
     a 10-second hiccup, checking DB in liveness will cause Kubernetes to restart
     all 20 backend pods simultaneously, causing a catastrophic thundering herd.

2. Readiness Probe (`/readyz`):
   - Question: "Is this pod ready to accept traffic right now?"
   - Kubernetes Action: If `/readyz` fails (e.g. Qdrant or Postgres is temporarily
     unreachable), Kubernetes REMOVES this pod from the Service load balancer.
     Incoming user requests stop routing to it until it recovers, WITHOUT restarting!

3. RED Metrics via Prometheus (`/metrics`):
   - Rate: Requests per second (`HTTP_REQUESTS_TOTAL`).
   - Errors: Failed requests per second (status codes >= 500).
   - Duration: Latency distribution (`HTTP_REQUEST_DURATION_SECONDS` histogram).
================================================================================
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

# Core RED metrics for all microservices across the platform
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
    """Discrete operational health statuses."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


HealthCheckFn = Callable[[], Awaitable[bool]]


class HealthCheckRegistry:
    """Registry for service dependencies and cloud-native readiness probes.

    Allows subsystems (e.g. Qdrant, Postgres, Redis) to register asynchronous
    connectivity health check lambdas that are evaluated during `/readyz`.
    """

    def __init__(self, service_name: str, version: str = "0.1.0") -> None:
        self.service_name = service_name
        self.version = version
        self._checks: dict[str, HealthCheckFn] = {}

    def register(self, name: str, check_fn: HealthCheckFn) -> None:
        """Register an async dependency check function (e.g. check_qdrant_ping).

        Args:
            name: Human-readable name of dependency (e.g. 'postgres', 'qdrant').
            check_fn: Async callable returning True if reachable, False otherwise.
        """
        self._checks[name] = check_fn

    async def liveness(self) -> JSONResponse:
        """Liveness probe: verifies process is running and event loop is responsive.

        Always returns HTTP 200 immediately unless the Python process is completely frozen.
        """
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
        """Readiness probe: validates all registered dependencies are accessible.

        Executes all registered dependency checks concurrently using `asyncio.gather`
        with a strict 3.0-second timeout per dependency to prevent probe hanging.
        """
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

        # Run all dependency checks in parallel
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
        """Generate Prometheus exposition format metrics for scraper ingestion."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
