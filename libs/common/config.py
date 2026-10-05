"""Base platform settings and configuration management."""

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

    # Core environment
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    APP_NAME: str = Field(default="ai-operations-platform", description="Platform application name")

    # API Gateway
    GATEWAY_HOST: str = Field(default="0.0.0.0", description="Gateway host binding")
    GATEWAY_PORT: int = Field(default=8000, description="Gateway port")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 prefix")
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    # PostgreSQL
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="aiops")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/aiops")

    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str = Field(default="")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # RabbitMQ
    RABBITMQ_HOST: str = Field(default="localhost")
    RABBITMQ_PORT: int = Field(default=5672)
    RABBITMQ_USER: str = Field(default="guest")
    RABBITMQ_PASSWORD: str = Field(default="guest")
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672/")

    # Qdrant Vector DB
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str = Field(default="")

    # Object Storage (MinIO / S3)
    S3_ENDPOINT_URL: str = Field(default="http://localhost:9000")
    S3_ACCESS_KEY_ID: str = Field(default="minioadmin")
    S3_SECRET_ACCESS_KEY: str = Field(default="minioadmin")
    S3_BUCKET_NAME: str = Field(default="aiops-storage")
    S3_REGION: str = Field(default="us-east-1")

    # OpenTelemetry & Observability
    OTEL_SERVICE_NAME: str = Field(default="aiops-gateway")
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None,
        description="OTLP collector gRPC endpoint, or None to disable remote export",
    )
    OTEL_EXPORTER_OTLP_INSECURE: bool = Field(default=True)
    PROMETHEUS_METRICS_PATH: str = Field(default="/metrics")

    # Langfuse Observability
    LANGFUSE_HOST: str = Field(default="http://localhost:3001")
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")

    # LLM Providers
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL_PRIMARY: str = Field(default="gpt-4o")
    OPENAI_MODEL_FAST: str = Field(default="gpt-4o-mini")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL_FALLBACK: str = Field(default="llama3.2:latest")

    # Security & Governance
    JWT_SECRET_KEY: str = Field(default="dev-secret-change-in-production-must-be-32-chars-min")
    JWT_ALGORITHM: str = Field(default="HS256")
    APPROVAL_TOKEN_SECRET: str = Field(default="dev-approval-secret-key-32-chars-long")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() == "testing"


@lru_cache
def get_settings() -> PlatformSettings:
    """Return cached instance of PlatformSettings."""
    return PlatformSettings()
