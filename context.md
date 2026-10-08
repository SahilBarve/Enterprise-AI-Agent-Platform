# Context — Enterprise AI Operations Platform

## Platform Summary & Core Architecture
A production-grade, multi-agent AI operations platform orchestrating 5 specialized agents under a LangGraph supervisor: Document RAG, Web Research, SQL Analytics, Data Processing, and Report Generation.
- **Tech Stack**: Python 3.11+, FastAPI, Pydantic v2, LangGraph, Qdrant, PostgreSQL (SQLAlchemy 2.0 + Alembic), Redis, RabbitMQ (aio-pika), LiteLLM, Docker, Kubernetes (EKS), Terraform, OpenTelemetry, Prometheus, Grafana.
- **P0 Non-Negotiable Pillars**:
  1. *MCP-first Tool Plane*: All tools/DB connections/search modules exposed via independent MCP servers; agents consume tools *only* through MCP client protocol.
  2. *Human-in-the-Loop Governance*: Risk-tiered, policy-driven approval checkpoints (`LangGraph.interrupt()` + Postgres checkpointer) for high-risk actions with HMAC-SHA256 signed single-use approval tokens bound to exact arguments.
  3. *Formal Evaluation*: Ragas + DeepEval evaluation harness for retrieval recall, faithfulness, trajectory efficiency, and SQL correctness, enforced as CI/CD regression gates.

- **GitHub Remote**: `https://github.com/SahilBarve/Enterprise-AI-Agent-Platform` (branch: `main`)

## Repository Structure (Planned Monorepo)
```
ai-ops-platform/
├── services/        # gateway, orchestrator, workers, ingestion, sandbox, scheduler, eval, mcp_servers
├── libs/            # common, llm, retrieval, agents, guardrails, tools, mcp_client
├── infra/           # docker, helm, terraform
├── frontend/        # web UI (Next.js)
├── observability/   # prometheus rules, grafana dashboards, otel collector configs
├── evals/           # golden datasets, synthetic generator, regression runners
├── tests/           # unit, integration, e2e, security, performance
├── scripts/         # dev tooling, seed data, bench runners
├── docker-compose.yml
└── Makefile
```

## Conventions & Standards
- **Naming**: snake_case for modules/functions, PascalCase for classes/types, UPPER_CASE for constants, kebab-case for directories/services.
- **Typing**: Strict type hints everywhere; Pydantic v2 for data transfer & config; mypy strict mode enforced.
- **Error Handling**: Standardized `AppError` hierarchy (`NotFoundError`, `ValidationError`, `AuthError`, `GovernanceError`, etc.) with RFC 7807 problem details and unique error codes.
- **Logging & Tracing**: Structured JSON logging (`structlog`); OpenTelemetry spans with correlation ID (`X-Correlation-ID`) propagated across HTTP, RabbitMQ headers, and MCP metadata.
- **Security & Tenancy**: Mandatory `tenant_id` at query and Qdrant payload levels; RBAC roles (`admin`, `analyst`, `viewer`, `service`); all external/retrieved content treated as untrusted (sanitization & prompt-injection defense).
- **Configuration**: `pydantic-settings` with `.env` and environment variables; zero hardcoded secrets or endpoints.

## How to Run & Test
- Local stack: `docker-compose up -d` (PostgreSQL, Redis, RabbitMQ, Qdrant, MinIO, OTel Collector, Prometheus, Grafana).
- Unit & integration tests: `pytest tests/unit tests/integration -v --cov=libs --cov=services`
- Linting & type checking: `ruff check .` && `ruff format --check .` && `mypy libs services`

## Delivery Roadmap & Phase Checklist
- [x] **Phase 0: Foundation** (Repo, tooling, config, docker-compose, libs/common, health/readiness, OTel base)
- [x] **Phase 1: Core RAG** (Ingestion, hybrid search Qdrant dense+sparse, reranker, citations, eval plumbing, gateway API)
- [x] **Phase 2: Optimization** (Two-tier semantic cache, context compression, token budgeter, adaptive CRAG, gateway integration)
- [x] **Phase 3: Orchestration & Governance** (LangGraph supervisor, SQL checkpointer, interrupt HITL, signed approval tokens, run lifecycle, crash recovery)
- [ ] **Phase 4: Specialist Agents & MCP Tool Plane** (Web, SQL, Data sandbox, Report agents as MCP servers)
- [ ] **Phase 5: Async & Scale** (RabbitMQ aio-pika workers, retries, DLQ, SSE progress, KEDA scaling)
- [ ] **Phase 6: Security & Guardrails** (RBAC, tenant isolation, input/output guardrails, audit logging)
- [ ] **Phase 7: Observability** (Grafana dashboards-as-code, Prometheus RED metrics, trace waterfalls, cost tracking)
- [ ] **Phase 8: Evaluation & CI Gates** (Golden datasets, Ragas/DeepEval metrics, CI regression blockers)
- [ ] **Phase 9: Cloud & MLOps** (Terraform AWS EKS, RDS, ElastiCache, Helm, blue/green re-indexing)
- [ ] **Phase 10: Frontend & Polish** (UI, Run Inspector, approval inbox, demo script, docs)
- [ ] **Phase 11: Stretch** (Playbooks, Slack bot, chaos tests)

## Current Status
- **Phase**: Phase 4 — Specialist Agents & MCP Tool Plane (Slice 4.1 Complete)
- **Done**:
  - Phases 1, 2, and 3 complete, tested, documented, and pushed to GitHub.
  - Slice 4.1: MCP Client Protocol & Core Tool Transport Plane ([`libs/mcp_client/`](file:///d:/Projects/AI-Operations-Platform/libs/mcp_client/)) with JSON-RPC 2.0 specifications, Stdio and In-Memory transports, schema validation, tool discovery, governance gate binding, and multi-server routing registry (FR-MCP-1, FR-MCP-2, FR-MCP-7).
  - 136 unit & integration tests passing with 92% coverage; strict Mypy and Ruff 100% green.
- **In Progress**: Slice 4.2: Web Research Agent & MCP Server (Tavily/DDG search, SSRF defenses, HTML scraper).
- **Next**: Slice 4.3 (SQL Analytics Agent & Safe Text-to-SQL MCP Server), Slice 4.4 (Data Processing Agent & Isolated Sandbox), Slice 4.5 (Report Generation Agent & End-to-End Exit Criteria).

## Known Gaps
- None.
