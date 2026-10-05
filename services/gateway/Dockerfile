# Multi-stage, minimal non-root Dockerfile for API Gateway
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir hatchling
COPY libs/ libs/
COPY services/ services/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd -r aiops && useradd -r -g aiops -d /app aiops \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY libs/ /app/libs/
COPY services/ /app/services/

USER aiops

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "services.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
