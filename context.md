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
- [ ] **Phase 2: Optimization** (Two-tier semantic cache, context compression, token budgeter, adaptive CRAG)
- [ ] **Phase 3: Orchestration & Governance** (LangGraph supervisor, Postgres checkpointer, interrupt HITL, signed approval tokens, run lifecycle)
- [ ] **Phase 4: Specialist Agents & MCP Tool Plane** (Web, SQL, Data sandbox, Report agents as MCP servers)
- [ ] **Phase 5: Async & Scale** (RabbitMQ aio-pika workers, retries, DLQ, SSE progress, KEDA scaling)
- [ ] **Phase 6: Security & Guardrails** (RBAC, tenant isolation, input/output guardrails, audit logging)
- [ ] **Phase 7: Observability** (Grafana dashboards-as-code, Prometheus RED metrics, trace waterfalls, cost tracking)
- [ ] **Phase 8: Evaluation & CI Gates** (Golden datasets, Ragas/DeepEval metrics, CI regression blockers)
- [ ] **Phase 9: Cloud & MLOps** (Terraform AWS EKS, RDS, ElastiCache, Helm, blue/green re-indexing)
- [ ] **Phase 10: Frontend & Polish** (UI, Run Inspector, approval inbox, demo script, docs)
- [ ] **Phase 11: Stretch** (Playbooks, Slack bot, chaos tests)

## Current Status
- **Phase**: Phase 2 — Optimization (Slice 2.3 Complete)
- **Done**:
  - Phase 1 Core RAG complete and baseline verified (Recall@10=1.0, MRR=1.0, Latency=1.28ms).
  - Codebase-wide educational documentation and mathematical formulas across all 21 modules.
  - Slice 2.1: Two-tier semantic cache ([`libs/retrieval/cache.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/cache.py)) with Tier 1 exact SHA-256 match, Tier 2 semantic embedding similarity ($\ge 0.92$), strict multi-tenant key isolation, cache poisoning protection (refusal & low confidence), TTL expiration, collection invalidation, and Prometheus hit/miss counters.
  - Slice 2.2: Context compressor & token budgeter ([`libs/retrieval/compressor.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/compressor.py)) with sentence-level relevance filtering, U-shaped "lost in the middle" attention reordering, token budget enforcement, and compression ratio reporting.
  - Slice 2.3: Adaptive Corrective RAG (CRAG) router ([`libs/retrieval/crag.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/crag.py)) with tri-state confidence grading (CORRECT $\ge 0.65$, AMBIGUOUS $\ge 0.30$, INCORRECT $< 0.30$), query rewriting/expansion, sub-query decomposition, and external fallback routing.
  - 93 unit & integration tests passing with 91% total coverage; strict Mypy (52 files) and Ruff 100% green.
- **In Progress**: Slice 2.4: Integration into RAG Generator & Gateway API; Phase 2 Exit Criteria verification.
- **Next**: Phase 3: Orchestration & Governance (LangGraph supervisor, Postgres checkpointer, approval tokens, HITL).

## Known Gaps
- None.
