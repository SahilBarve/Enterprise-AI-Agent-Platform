"""Base platform settings and configuration management.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Pydantic BaseSettings for 12-Factor App Configuration?
--------------------------------------------------------------------------------
The 12-Factor App methodology dictates: "Store config in the environment."
In enterprise platforms spanning multiple environments (local dev, CI/CD, staging,
production Kubernetes), configuration must be decoupled from code:

1. Type Safety & Validation:
   Standard Python `os.getenv("PORT")` returns a string. If someone sets `PORT=foo`,
   your service crashes silently later. Pydantic Settings validates types at startup:
   e.g. `GATEWAY_PORT: int` ensures that invalid ports fail immediately on launch.

2. Zero Hardcoded Secrets:
   Every secret (JWT keys, DB passwords, S3 credentials) defaults to local development
   placeholders but can be seamlessly overridden in production via Kubernetes Secrets
   or AWS Secrets Manager environment variables.

3. Singleton Caching via `@lru_cache`:
   Loading `.env` files from disk has I/O overhead. Decorating `get_settings()`
   with `@lru_cache` ensures configuration is parsed once and cached in memory.
================================================================================
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    """Central platform settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core environment and runtime flags
    ENVIRONMENT: str = Field(default="development", description="Runtime environment: development, testing, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity level: DEBUG, INFO, WARNING, ERROR")
    APP_NAME: str = Field(default="ai-operations-platform", description="Platform application identifier")

    # API Gateway configuration
    GATEWAY_HOST: str = Field(default="0.0.0.0", description="Gateway IP address binding")
    GATEWAY_PORT: int = Field(default=8000, description="Gateway HTTP port")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API route prefix for version 1")
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for web browser clients",
    )

    # PostgreSQL Relational Storage (Audit logs, runs, checkpoints)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="aiops")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/aiops",
        description="Async SQLAlchemy database connection string",
    )

    # Redis In-Memory Cache (Semantic cache, session state, rate limits)
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str = Field(default="")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # RabbitMQ Asynchronous Task Broker
    RABBITMQ_HOST: str = Field(default="localhost")
    RABBITMQ_PORT: int = Field(default=5672)
    RABBITMQ_USER: str = Field(default="guest")
    RABBITMQ_PASSWORD: str = Field(default="guest")
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672/")

    # Qdrant Vector Search Engine
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str = Field(default="")

    # Object Storage (MinIO local / AWS S3)
    S3_ENDPOINT_URL: str = Field(default="http://localhost:9000")
    S3_ACCESS_KEY_ID: str = Field(default="minioadmin")
    S3_SECRET_ACCESS_KEY: str = Field(default="minioadmin")
    S3_BUCKET_NAME: str = Field(default="aiops-storage")
    S3_REGION: str = Field(default="us-east-1")

    # OpenTelemetry & Observability Exporters
    OTEL_SERVICE_NAME: str = Field(default="aiops-gateway")
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None,
        description="OTLP collector gRPC endpoint, or None to disable remote export",
    )
    OTEL_EXPORTER_OTLP_INSECURE: bool = Field(default=True)
    PROMETHEUS_METRICS_PATH: str = Field(default="/metrics")

    # Langfuse Tracing & Evaluation Platform
    LANGFUSE_HOST: str = Field(default="http://localhost:3001")
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")

    # LLM Providers (OpenAI & Local Ollama fallback)
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL_PRIMARY: str = Field(default="gpt-4o")
    OPENAI_MODEL_FAST: str = Field(default="gpt-4o-mini")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL_FALLBACK: str = Field(default="llama3.2:latest")

    # Security & Governance Gate Secrets
    JWT_SECRET_KEY: str = Field(default="dev-secret-change-in-production-must-be-32-chars-min")
    JWT_ALGORITHM: str = Field(default="HS256")
    APPROVAL_TOKEN_SECRET: str = Field(default="dev-approval-secret-key-32-chars-long")

    @property
    def is_production(self) -> bool:
        """Helper to check if active runtime environment is production."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_testing(self) -> bool:
        """Helper to check if active runtime environment is pytest/testing."""
        return self.ENVIRONMENT.lower() == "testing"


@lru_cache
def get_settings() -> PlatformSettings:
    """Return cached singleton instance of PlatformSettings.

    Can be cleared in unit tests using get_settings.cache_clear().
    """
    return PlatformSettings()
