"""API Gateway service entrypoint and router configuration.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why the Application Factory Pattern and RFC 7807 in FastAPI?
--------------------------------------------------------------------------------
1. Application Factory (`create_gateway_app`):
   Rather than instantiating a single global `app = FastAPI()` at module level,
   wrapping creation inside a function allows tests to inject custom `PlatformSettings`,
   mock database connections, or disable external exporters without polluting
   the global Python environment.

2. Middleware Order Matters:
   - `CorrelationIdMiddleware` is mounted FIRST. It intercepts the HTTP request,
     extracts or generates `X-Correlation-ID`, and sets it in Python's `contextvars`.
     Because it runs first, any log message or OpenTelemetry span created by later
     middleware or route handlers automatically contains the correlation ID!
   - `CORSMiddleware` handles browser preflight (`OPTIONS`) requests.

3. RFC 7807 Problem Details:
   Microservices should never return raw unhandled traceback strings to clients.
   RFC 7807 is the IETF standard for HTTP error responses. The custom exception
   handlers convert both domain exceptions (`AppError`) and FastAPI schema errors
   (`RequestValidationError`) into standardized JSON:
   {
       "type": "urn:aiops:error:not-found",
       "title": "RESOURCE_NOT_FOUND",
       "status": 404,
       "detail": "Collection runbooks does not exist",
       "instance": "/api/v1/collections/runbooks/documents"
   }

4. Cloud-Native Probes:
   - `/healthz` (Liveness): Tells Kubernetes whether the container process is alive.
   - `/readyz` (Readiness): Tells Kubernetes whether the service is ready to receive
     traffic (e.g. Qdrant is connected, Postgres is healthy).
   - `/metrics` (Prometheus): Exposes RED metrics (Rate, Errors, Duration) for Grafana.
================================================================================
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response

from libs.common.config import PlatformSettings, get_settings
from libs.common.errors import AppError
from libs.common.health import HealthCheckRegistry
from libs.common.logging import configure_logging, get_logger
from libs.common.telemetry import CorrelationIdMiddleware, setup_telemetry
from services.gateway.rag_router import router as rag_router

logger = get_logger("gateway")


def create_gateway_app(settings: PlatformSettings | None = None) -> FastAPI:
    """Application factory for the API Gateway service.

    Args:
        settings: Optional custom platform settings (useful for test overrides).

    Returns:
        Configured FastAPI application instance.
    """
    active_settings = settings or get_settings()

    # Configure structured logging and OTel telemetry
    configure_logging(log_level=active_settings.LOG_LEVEL)
    otlp_endpoint = (
        None if active_settings.is_testing else active_settings.OTEL_EXPORTER_OTLP_ENDPOINT
    )
    setup_telemetry(
        service_name=active_settings.OTEL_SERVICE_NAME,
        otlp_endpoint=otlp_endpoint,
        insecure=active_settings.OTEL_EXPORTER_OTLP_INSECURE,
    )

    health_registry = HealthCheckRegistry(
        service_name="api-gateway",
        version="0.1.0",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        """Modern ASGI lifespan handler managing startup and graceful shutdown."""
        logger.info(
            "API Gateway starting",
            environment=active_settings.ENVIRONMENT,
            port=active_settings.GATEWAY_PORT,
        )
        yield
        logger.info("API Gateway shutting down")

    app = FastAPI(
        title="Enterprise AI Operations Platform API Gateway",
        description="Public API Gateway for multi-agent workflows, RAG, and governance",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware 1: Correlation ID (first in chain so context is available to all components)
    app.add_middleware(CorrelationIdMiddleware)

    # Middleware 2: Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handler: Domain / AppError -> RFC 7807 Problem Details
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Translate platform domain exceptions to RFC 7807 problem details."""
        logger.warning(
            "Application error caught",
            error_code=exc.error_code,
            status_code=exc.status_code,
            detail=exc.message,
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_problem_detail(instance_path=str(request.url.path)),
        )

    # Exception Handler: RequestValidationError -> RFC 7807 Problem Details
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Translate Pydantic schema validation errors to RFC 7807 problem details."""
        logger.info(
            "Validation error on request",
            path=str(request.url.path),
            errors=exc.errors(),
        )
        problem = {
            "type": "urn:aiops:error:validation-error",
            "title": "VALIDATION_ERROR",
            "status": 422,
            "detail": "Input data failed schema validation",
            "error_code": "VALIDATION_ERROR",
            "invalid_params": exc.errors(),
            "instance": str(request.url.path),
        }
        return JSONResponse(status_code=422, content=problem)

    # Health and Observability Endpoints (FR-GW-8)
    @app.get("/healthz", tags=["System"], summary="Liveness probe")
    async def healthz() -> JSONResponse:
        """Liveness check: returns 200 if process is up."""
        return await health_registry.liveness()

    @app.get("/readyz", tags=["System"], summary="Readiness probe")
    async def readyz() -> JSONResponse:
        """Readiness check: returns 200 if all dependent subsystems are operational."""
        return await health_registry.readiness()

    @app.get("/metrics", tags=["System"], summary="Prometheus metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint in OpenMetrics exposition format."""
        return health_registry.metrics()

    # Base info endpoint
    @app.get(active_settings.API_V1_PREFIX + "/status", tags=["Status"])
    async def status() -> dict[str, str]:
        """Base service info and operational status endpoint."""
        return {
            "app": active_settings.APP_NAME,
            "status": "online",
            "version": "0.1.0",
            "environment": active_settings.ENVIRONMENT,
        }

    # Register domain routers
    app.include_router(rag_router)

    return app


# Default ASGI application instance for uvicorn entrypoint
app = create_gateway_app()
