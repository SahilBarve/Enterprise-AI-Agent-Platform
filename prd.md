# PRD — Enterprise AI Operations Platform

**Version:** 1.0 | **Owner:** Sahil | **Status:** Ready for implementation

> Priority legend: **P0** = must ship (core resume claims), **P1** = should ship (strong differentiators), **P2** = stretch (polish / wow factor).

---

## 1. Overview

### 1.1 Vision
A production-grade, multi-agent AI operations platform that lets an organization ask questions or assign tasks in natural language, and have autonomous agents carry out the work end-to-end: searching internal documents, researching the web, querying databases, processing data, and producing finished reports, with every step observable, evaluated, secured, and auditable.

### 1.2 Problem Statement
Teams lose time switching between document search, BI dashboards, spreadsheets, web research, and report writing. Single-LLM chatbots hallucinate, cannot act, and are not production-ready (no tracing, cost control, access control, or evaluation). This platform replaces that with orchestrated, specialized agents running on real distributed infrastructure.

### 1.3 Goals
1. Autonomous multi-step workflows across five capability domains (Doc RAG, Web Research, SQL Analytics, Data Processing, Report Generation).
2. A best-practice RAG pipeline: hybrid retrieval, cross-encoder reranking, semantic caching, context compression.
3. Production infrastructure: microservices, Docker, Kubernetes, CI/CD, AWS, async task execution, full observability.
4. Trust: citations, groundedness checks, guardrails, human approval for risky actions, audit logs.
5. Measurable quality: automated evaluation harness with regression gates in CI.

### 1.4 Non-Goals
- Training or fine-tuning foundation models (only optional fine-tuning of embedding/reranker is a P2 stretch).
- Building a general-purpose BI tool or a full data warehouse.
- Real-time voice or video interfaces.

### 1.5 Personas
| Persona | Needs |
|---|---|
| **Business Analyst** | Ask data questions in English, get SQL-backed charts and reports |
| **Operations Manager** | Scheduled summary reports, approvals for sensitive actions |
| **Knowledge Worker** | Search across thousands of internal documents with cited answers |
| **Platform Admin** | Manage tenants, users, quotas, cost, model config, system health |
| **ML/AI Engineer** | Trace agent runs, evaluate quality, version prompts, debug failures |

### 1.6 Success Metrics
| Metric | Target |
|---|---|
| Retrieval Recall@10 on golden set | ≥ 0.90 |
| Answer faithfulness (RAGAS) | ≥ 0.85 |
| Answer relevancy (RAGAS) | ≥ 0.85 |
| Semantic cache hit rate (steady state) | ≥ 30% |
| Context compression ratio | ≥ 40% token reduction with < 3% quality drop |
| p95 latency, simple RAG query (cache miss) | < 6 s |
| p95 latency, cache hit | < 400 ms |
| Text-to-SQL execution accuracy on test set | ≥ 80% |
| API availability (staging SLO) | 99.5% |
| CI pipeline duration | < 12 min |

---

## 2. System Architecture

### 2.1 High-Level Diagram
```
                    ┌──────────────────────────────┐
                    │  Web UI (React/Next.js) / API │
                    └──────────────┬───────────────┘
                                   │ HTTPS / SSE / WebSocket
                    ┌──────────────▼───────────────┐
                    │ API Gateway Service (FastAPI) │  auth, rate limit, RBAC,
                    │                               │  request validation, tracing
                    └───────┬──────────────┬────────┘
                            │              │
              ┌─────────────▼───┐    ┌─────▼───────────────┐
              │ Orchestrator Svc │    │ Ingestion Service    │
              │ (LangGraph)      │    │ (parse/chunk/embed)  │
              └───┬──────────────┘    └─────┬───────────────┘
                  │ RabbitMQ (task queues)   │
   ┌──────────────┼──────────┬───────────────┼──────────────┐
   ▼              ▼          ▼               ▼              ▼
┌────────┐  ┌──────────┐ ┌─────────┐  ┌────────────┐ ┌────────────┐
│Doc RAG │  │Web Resrch│ │SQL Agent│  │Data Proc   │ │Report Gen  │
│ Agent  │  │ Agent    │ │         │  │Agent(sandbx)│ │Agent       │
└───┬────┘  └────┬─────┘ └────┬────┘  └─────┬──────┘ └─────┬──────┘
    │            │            │             │              │
 Qdrant      Search APIs   PostgreSQL   S3 / Sandbox   S3 / Templates
 Redis (cache, rate limit, sessions, checkpoints locks)
 PostgreSQL (metadata, checkpoints, audit, usage, prompts)
 Prometheus + Grafana + OpenTelemetry Collector (+ Tempo/Jaeger, Loki)
```

### 2.2 Services
| Service | Responsibility |
|---|---|
| `gateway` | Public API, auth, RBAC, rate limiting, request routing, SSE streaming |
| `orchestrator` | LangGraph supervisor graph, planning, agent routing, state & checkpoints |
| `agent-workers` | Celery/aio-pika consumers executing agent tasks per queue |
| `ingestion` | Document upload, parsing, chunking, embedding, indexing |
| `retrieval` | Hybrid retrieval, reranking, cache, compression (shared library + optional service) |
| `sandbox` | Isolated code execution for data processing |
| `scheduler` | Cron-based recurring workflows and reports |
| `eval` | Offline/online evaluation jobs and golden dataset runner |
| `frontend` | Web UI |

### 2.3 Tech Stack
| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI, Pydantic v2, Uvicorn/Gunicorn |
| Agent orchestration | LangGraph (+ Postgres checkpointer), LangChain core utilities |
| LLM access | Provider-agnostic layer via LiteLLM (cloud models + Ollama local fallback) |
| Vector DB | Qdrant (dense + sparse vectors, payload filtering) |
| Relational DB | PostgreSQL (SQLAlchemy 2.0 + Alembic migrations) |
| Cache / state | Redis |
| Message broker | RabbitMQ (+ Celery or aio-pika workers) |
| Object storage | AWS S3 (MinIO locally) |
| Embeddings | BGE-M3 or bge-base-en-v1.5 (configurable) |
| Reranker | bge-reranker-v2-m3 or ms-marco MiniLM cross-encoder (configurable) |
| Containerization | Docker (multi-stage builds), Docker Compose for local |
| Orchestration | Kubernetes (EKS), Helm charts |
| IaC | Terraform (VPC, EKS, RDS, ElastiCache, S3, ECR, IAM) |
| CI/CD | GitHub Actions |
| Observability | OpenTelemetry, Prometheus, Grafana, Loki/Tempo (or Jaeger), Langfuse (optional LLM tracing) |
| Frontend | Next.js/React + Tailwind (or Streamlit for MVP) |
| Testing | pytest, pytest-asyncio, testcontainers, Locust, RAGAS |

---

## 3. Functional Requirements

### 3.1 API Gateway & Platform Services (P0)
- **FR-GW-1** REST API with OpenAPI docs; versioned under `/api/v1`.
- **FR-GW-2** JWT auth (access + refresh tokens), API keys for service clients, password hashing (argon2/bcrypt).
- **FR-GW-3** Multi-tenancy: every resource (documents, runs, reports, connections) is scoped by `tenant_id`; enforced at the query layer and in Qdrant payload filters.
- **FR-GW-4** RBAC roles: `admin`, `analyst`, `viewer`, `service`. Permission matrix documented and tested.
- **FR-GW-5** Rate limiting per user/tenant/API key (Redis token bucket).
- **FR-GW-6** Streaming responses via SSE (token streaming + agent step events).
- **FR-GW-7** Idempotency keys on task-creating endpoints.
- **FR-GW-8** Health (`/healthz`), readiness (`/readyz`), and metrics (`/metrics`) endpoints on every service.
- **FR-GW-9** Request correlation ID propagated through all services, queues, and logs.
- **FR-GW-10** Standardized error model and typed exceptions.

### 3.2 Multi-Agent Orchestration (P0)
- **FR-OR-1** Supervisor graph built with LangGraph that receives a user goal, classifies intent, creates a **plan** (ordered/parallel steps), and delegates to specialized agents.
- **FR-OR-2** Agents are LangGraph subgraphs with typed state; shared state schema with reducers.
- **FR-OR-3** Supported flows: single-agent, sequential multi-agent, parallel fan-out/fan-in, conditional branching, and loops with max-iteration guards.
- **FR-OR-4** **Planner → Executor → Critic** loop: a critic node validates each result (completeness, grounding, format) and can trigger a retry or re-plan.
- **FR-OR-5** Persistent checkpointing (Postgres) so runs survive crashes, can be resumed, paused, and cancelled.
- **FR-OR-6** **Human-in-the-loop** via LangGraph interrupts: approval required for configured actions (running write SQL, sending emails/webhooks, executing generated code on large data, expensive web crawls).
- **FR-OR-7** Per-run budgets: max steps, max tokens, max cost, max wall-clock time; graceful termination with partial results.
- **FR-OR-8** Tool registry with JSON-schema-defined tools, per-agent tool allowlists, and per-tenant tool enablement.
- **FR-OR-9** Run lifecycle states: `queued → planning → running → awaiting_approval → completed | failed | cancelled`.
- **FR-OR-10** Run replay and **time-travel debugging**: fork a run from any checkpoint with edited state or a different prompt/model (P1).
- **FR-OR-11** Long-term memory: user/tenant-level preferences and learned facts stored with embeddings, retrievable by the planner (P1).
- **FR-OR-12** Conversation threads with short-term memory and automatic summarization when context grows large.

### 3.3 Document RAG Agent (P0)

#### 3.3.1 Ingestion
- **FR-RAG-1** Upload via API/UI: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, and images (OCR) ; bulk upload and zip upload.
- **FR-RAG-2** Pluggable connectors (P1): S3 bucket sync, Google Drive, Confluence/Notion, URL crawl.
- **FR-RAG-3** Async ingestion pipeline via RabbitMQ with status tracking (`uploaded → parsing → chunking → embedding → indexed | failed`) and retry with dead-letter queue.
- **FR-RAG-4** Layout-aware parsing (Unstructured / PyMuPDF / Docling) preserving headings, tables, page numbers; OCR fallback for scans (Tesseract/PaddleOCR).
- **FR-RAG-5** Chunking strategies (configurable per collection): recursive, semantic, heading-aware, and **parent-child (small-to-big)** chunking. Table-aware chunking that keeps table rows with headers.
- **FR-RAG-6** Metadata extraction: title, author, doc type, dates, page, section path, language, tenant, ACL tags, content hash.
- **FR-RAG-7** Deduplication by content hash and near-duplicate detection (MinHash/SimHash); versioning of re-uploaded documents.
- **FR-RAG-8** Optional contextual chunk enrichment: LLM-generated chunk summary/context prefix before embedding (contextual retrieval) (P1).
- **FR-RAG-9** PII detection at ingest with configurable redaction or tagging (P1).
- **FR-RAG-10** Document lifecycle: delete, re-index, and collection management with cascading cleanup of vectors and S3 objects.

#### 3.3.2 Retrieval
- **FR-RAG-11** **Hybrid retrieval**: dense vectors + sparse/BM25 vectors in Qdrant, merged with Reciprocal Rank Fusion (weights configurable).
- **FR-RAG-12** **Query understanding**: query rewriting, multi-query expansion, HyDE (hypothetical document embeddings), decomposition of complex questions into sub-queries (each toggleable).
- **FR-RAG-13** Metadata/ACL filtering applied pre-retrieval (tenant, doc type, date range, tags, user permissions).
- **FR-RAG-14** **Cross-encoder reranking** of top-K candidates (e.g., 50 → 8) with batched inference and score thresholds.
- **FR-RAG-15** MMR / diversity re-selection to avoid redundant chunks.
- **FR-RAG-16** Parent-document expansion: retrieve small chunks, return surrounding/parent context.
- **FR-RAG-17** **Adaptive/Corrective RAG**: a retrieval grader scores relevance; if low, the agent rewrites the query, widens search, or falls back to web research (Self-RAG / CRAG pattern) (P1).

#### 3.3.3 Semantic Caching
- **FR-RAG-18** Two-tier cache in Redis: exact-match (normalized query hash) and **semantic** (embedding similarity above threshold, e.g., 0.92).
- **FR-RAG-19** Cache key includes tenant, ACL scope, collection version, model, and prompt version to prevent leaks and stale answers.
- **FR-RAG-20** TTL + event-based invalidation when underlying documents change; cache hit/miss/stale metrics.
- **FR-RAG-21** Admin controls: per-tenant enable/disable, threshold tuning, manual flush.
- **FR-RAG-22** Cache poisoning protections (do not cache failed/low-confidence/refused answers).

#### 3.3.4 Context Compression
- **FR-RAG-23** Compression pipeline before generation: sentence-level relevance filtering, extractive compression, and optional LLM/LLMLingua-style token pruning.
- **FR-RAG-24** Token budget manager that fits context to the model window with priority ordering and reports compression ratio per request.
- **FR-RAG-25** "Lost in the middle" mitigation: reorder chunks so highest-relevance items sit at the start and end.

#### 3.3.5 Generation & Grounding
- **FR-RAG-26** Answers include **inline citations** mapping to document, page, and chunk, with click-through to the source passage in the UI.
- **FR-RAG-27** **Groundedness/hallucination check**: claim-level verification of the answer against retrieved context (NLI or LLM judge); unsupported claims flagged or removed; confidence score returned.
- **FR-RAG-28** "I don't know" behavior when evidence is insufficient.
- **FR-RAG-29** Structured output modes (JSON schema) for downstream agents.

### 3.4 Web Research Agent (P0)
- **FR-WEB-1** Search via pluggable providers (Tavily/SerpAPI/Brave/DuckDuckGo) with provider fallback.
- **FR-WEB-2** Page fetching with robots.txt respect, timeouts, retries, domain allow/deny lists, and SSRF protection (block private IP ranges).
- **FR-WEB-3** Content extraction (readability/trafilatura), cleaning, chunking, and on-the-fly reranking.
- **FR-WEB-4** Iterative research loop: generate sub-questions → search → read → synthesize → identify gaps → repeat up to a depth limit ("deep research" mode).
- **FR-WEB-5** Source credibility scoring (domain reputation, recency, corroboration across sources); contradictions highlighted.
- **FR-WEB-6** Citations with URL, title, access timestamp; optional page snapshot archiving to S3 (P2).
- **FR-WEB-7** Result caching in Redis with TTL; rate limiting and politeness delays.
- **FR-WEB-8** Prompt-injection defenses on fetched content (content treated as untrusted data; instruction-stripping and isolation).

### 3.5 SQL Analytics Agent (P0)
- **FR-SQL-1** Data source management: register PostgreSQL (and optionally MySQL/SQLite/Snowflake) connections with encrypted credentials; read-only DB role enforced.
- **FR-SQL-2** **Schema introspection and semantic layer**: tables, columns, types, FK relationships, sample values, and human-written descriptions; stored and embedded for **schema linking** (retrieve only relevant tables/columns per question).
- **FR-SQL-3** Text-to-SQL pipeline: schema linking → SQL generation → static validation → dry-run (`EXPLAIN`) → execution → result validation → natural-language explanation.
- **FR-SQL-4** **SQL safety**: parse with `sqlglot`/AST; allow only `SELECT` (configurable); block DDL/DML, multiple statements, dangerous functions; enforce `LIMIT`, statement timeout, row cap, and cost threshold via `EXPLAIN`.
- **FR-SQL-5** **Self-correction loop**: on SQL error or empty/implausible result, feed error back to the model for repair (bounded retries).
- **FR-SQL-6** Few-shot example store: successful (question, SQL) pairs retrieved by similarity to improve future accuracy; user thumbs-up promotes a pair to the store.
- **FR-SQL-7** Business glossary & metric definitions (e.g., "active customer", "MRR") applied during generation.
- **FR-SQL-8** Result presentation: table, auto-selected chart type (Plotly/Vega), summary insights, and the exact SQL shown for transparency.
- **FR-SQL-9** Row/column-level security: tenant-scoped views, column masking for PII.
- **FR-SQL-10** Query history, saved queries, and exportable results (CSV/XLSX).
- **FR-SQL-11** Ambiguity handling: ask clarifying questions when the request maps to multiple interpretations (P1).

### 3.6 Data Processing Agent (P0)
- **FR-DP-1** Accept CSV/XLSX/Parquet/JSON uploads or SQL results as inputs.
- **FR-DP-2** Automated data profiling: schema, null rates, distributions, outliers, duplicates, correlations.
- **FR-DP-3** Cleaning & transformation: type coercion, missing-value strategies, dedup, normalization, joins/pivots, feature derivation — generated as reviewable pandas/Polars code.
- **FR-DP-4** **Sandboxed code execution**: generated Python runs in an isolated container (no network, CPU/memory/time limits, read-only FS except scratch, allowlisted libraries); implemented with Docker/gVisor/Firecracker-style isolation or a K8s Job with strict `securityContext`.
- **FR-DP-5** Code review step: static analysis (AST allowlist/bandit) before execution; optional human approval.
- **FR-DP-6** Statistical analysis & light ML: descriptive stats, hypothesis tests, trend/seasonality, anomaly detection, forecasting, clustering.
- **FR-DP-7** Output artifacts (cleaned datasets, charts, notebooks) stored in S3 with lineage metadata.
- **FR-DP-8** **Data lineage**: every output records input datasets, code version, parameters, and run ID; reproducible re-run.
- **FR-DP-9** Large file handling via chunked/streaming processing and async jobs with progress events.

### 3.7 Report Generation Agent (P0)
- **FR-REP-1** Generate structured reports from outputs of other agents: executive summary, findings, charts, tables, methodology, citations, and appendices.
- **FR-REP-2** Output formats: Markdown, HTML, **PDF**, **DOCX**, **PPTX** (P1), and JSON.
- **FR-REP-3** Template system (Jinja2) with tenant branding (logo, colors, fonts); user-defined report templates.
- **FR-REP-4** Section-by-section generation with outline approval step (HITL optional) and per-section regeneration.
- **FR-REP-5** Embedded charts (Plotly/Matplotlib → images) and data tables with source tracing for every number ("click a figure → see the SQL/document/source").
- **FR-REP-6** Report versioning, diff between versions, and sharing via expiring signed links.
- **FR-REP-7** **Scheduled/recurring reports** (daily/weekly/monthly) with email/Slack/webhook delivery (P1).
- **FR-REP-8** Quality pass: fact-check against sources, consistency check (numbers match across sections), and style/tone control.

### 3.8 Async Task Execution (P0)
- **FR-ASYNC-1** All long-running work (ingestion, agent runs, code execution, report generation) dispatched through RabbitMQ with dedicated queues per workload class and priority queues.
- **FR-ASYNC-2** Reliable delivery: durable queues, manual acks, retries with exponential backoff + jitter, **dead-letter queues**, poison-message handling.
- **FR-ASYNC-3** Idempotent consumers, deduplication keys, and outbox pattern for DB-to-queue consistency (P1).
- **FR-ASYNC-4** Real-time progress to clients via Redis pub/sub → SSE/WebSocket.
- **FR-ASYNC-5** Task cancellation, timeout, and visibility into queue depth.
- **FR-ASYNC-6** Worker autoscaling driven by queue depth (KEDA) (P1).

### 3.9 Security, Guardrails & Governance (P0/P1)
- **FR-SEC-1** (P0) Secrets in AWS Secrets Manager / K8s Secrets (External Secrets); no secrets in images or repo; secret scanning in CI.
- **FR-SEC-2** (P0) TLS everywhere; network policies in K8s; least-privilege IAM (IRSA).
- **FR-SEC-3** (P0) **Input guardrails**: prompt-injection/jailbreak detection, input length limits, PII detection.
- **FR-SEC-4** (P0) **Output guardrails**: PII leakage filter, toxicity/policy filter, schema validation, groundedness enforcement.
- **FR-SEC-5** (P0) Tool-use permissioning and sandboxing; indirect prompt-injection mitigation (untrusted content isolation).
- **FR-SEC-6** (P0) **Audit log**: immutable record of who ran what, tools called, data accessed, approvals granted.
- **FR-SEC-7** (P1) Data retention policies and tenant data export/delete (GDPR-style).
- **FR-SEC-8** (P1) Dependency and container vulnerability scanning (Trivy, pip-audit), SBOM generation, image signing (cosign).
- **FR-SEC-9** (P1) OWASP Top 10 for LLM Applications checklist mapped to implemented controls.

### 3.10 Observability & Monitoring (P0)
- **FR-OBS-1** **OpenTelemetry** instrumentation across all services: traces spanning gateway → queue → worker → LLM/tool calls; context propagated via message headers.
- **FR-OBS-2** **Prometheus metrics**: request rate/latency/errors (RED), queue depth/consumer lag, cache hit ratio, retrieval latency by stage, reranker latency, token usage, cost, agent step counts, tool error rates, sandbox failures.
- **FR-OBS-3** **Grafana dashboards** (provisioned as code): Platform Overview, RAG Quality & Latency, Agent Runs, Queue Health, LLM Cost & Tokens, Infrastructure (K8s), SLO dashboard.
- **FR-OBS-4** Structured JSON logging with trace/run IDs; centralized logs (Loki/CloudWatch).
- **FR-OBS-5** Alerting (Alertmanager): high error rate, SLO burn, queue backlog, DLQ growth, cache hit collapse, cost spike, pod crash loops.
- **FR-OBS-6** LLM-specific tracing: prompts, completions, tool I/O, latencies, token counts per step (Langfuse or custom), with redaction of sensitive data.
- **FR-OBS-7** Per-tenant usage and **cost tracking** (tokens × model price) with budgets and alerts.

### 3.11 Evaluation & Quality (P1 — key differentiator)
- **FR-EVAL-1** Golden datasets (questions, expected answers, supporting docs, expected SQL) stored in repo/DB; synthetic test-set generation from ingested documents.
- **FR-EVAL-2** Retrieval metrics: Recall@k, MRR, nDCG, hit rate; compare configurations (dense-only vs hybrid vs hybrid+rerank).
- **FR-EVAL-3** Generation metrics via RAGAS/DeepEval: faithfulness, answer relevancy, context precision/recall; LLM-as-judge with calibrated rubrics.
- **FR-EVAL-4** SQL execution-accuracy evaluation against reference results.
- **FR-EVAL-5** Agent trajectory evaluation: correct tool selection, step efficiency, success rate, cost per task.
- **FR-EVAL-6** **CI evaluation gate**: PRs that regress metrics beyond thresholds fail the pipeline; results posted as PR comments.
- **FR-EVAL-7** Online feedback: thumbs up/down, comments, and failure tagging feeding back into eval sets and few-shot stores.
- **FR-EVAL-8** Published **benchmark report** in the repo comparing ablations (this is a portfolio highlight).

### 3.12 Model & Prompt Management (P1)
- **FR-MOD-1** Provider-agnostic LLM layer with retries, timeouts, fallbacks, and circuit breakers.
- **FR-MOD-2** **Model router**: cheap/fast model for classification, rewriting, and simple queries; stronger model for planning and synthesis; local Ollama model as fallback/offline mode; routing rules configurable.
- **FR-MOD-3** **Prompt registry**: versioned prompts in DB/files with metadata, A/B testing, canary rollout, and rollback; prompt version recorded on every trace.
- **FR-MOD-4** Structured output enforcement (JSON schema / function calling) with automatic repair on parse failure.
- **FR-MOD-5** Token counting, context-window management, and streaming support.

### 3.13 Web Frontend (P0/P1)
- **FR-UI-1** (P0) Chat/workspace UI with streaming answers, citations, and live agent-step timeline.
- **FR-UI-2** (P0) Document library: upload, status, search, delete, collection management.
- **FR-UI-3** (P0) Data source & SQL workspace: schema browser, query results, charts.
- **FR-UI-4** (P0) Reports page: generate, preview, download, version history.
- **FR-UI-5** (P1) **Run Inspector / Trace Viewer**: visual graph of the agent run (nodes, tool calls, latencies, tokens, cost), with state at each checkpoint and "fork from here".
- **FR-UI-6** (P1) Approval inbox for human-in-the-loop requests.
- **FR-UI-7** (P1) Admin console: users, roles, tenants, quotas, model/prompt settings, cache controls, usage and cost.
- **FR-UI-8** (P1) Evaluation dashboard: metric trends, regression diffs.
- **FR-UI-9** (P2) Workflow builder: save and re-run reusable multi-agent workflows ("playbooks") with parameters.

### 3.14 Developer Experience & Integrations (P1/P2)
- **FR-DX-1** (P1) Python SDK and CLI (`aiops run "..."`, `aiops ingest ./docs`).
- **FR-DX-2** (P1) Webhooks for run/report events.
- **FR-DX-3** (P2) **MCP (Model Context Protocol) support**: expose platform tools as an MCP server and consume external MCP tools.
- **FR-DX-4** (P2) Slack/Teams bot interface.
- **FR-DX-5** (P2) Example "playbooks": Weekly Sales Report, Competitor Brief, Contract Q&A, Data Quality Audit.

---

## 4. Standout Features (Differentiators)

These go beyond a standard RAG/agent project and are the reason this platform should stand out in interviews and on a resume:

| # | Feature | Why it stands out | Priority |
|---|---|---|---|
| 1 | **Evaluation harness + CI regression gate** (RAGAS, retrieval metrics, SQL accuracy, trajectory eval) | Most projects have zero measurement; this proves engineering rigor | P1 |
| 2 | **Planner–Executor–Critic loop with self-correction** | Shows real agentic design, not a linear chain | P0 |
| 3 | **Human-in-the-loop approvals** via LangGraph interrupts | Production safety pattern enterprises require | P0/P1 |
| 4 | **Time-travel debugging & run forking** from checkpoints | Rare, highly demonstrable feature | P1 |
| 5 | **Run Inspector UI** (trace graph, cost, tokens per node) | Makes the system's intelligence visible in demos | P1 |
| 6 | **Corrective/Adaptive RAG** with web fallback | Advanced retrieval behavior tied to agent routing | P1 |
| 7 | **Claim-level groundedness verification + citations** | Directly attacks hallucination | P0/P1 |
| 8 | **Safe Text-to-SQL** (AST validation, EXPLAIN cost gate, semantic layer, few-shot memory) | Realistic enterprise-grade analytics | P0 |
| 9 | **Sandboxed code execution** for data agent | Security depth, hard to do well | P0 |
| 10 | **Model router + fallback + cost budgets** | Real cost-engineering story with measurable savings | P1 |
| 11 | **Prompt registry with A/B tests and canary rollout** | LLMOps maturity | P1 |
| 12 | **Guardrails suite** (prompt injection, PII, output policy) + OWASP-LLM mapping | Security-forward AI engineering | P0/P1 |
| 13 | **Queue-depth autoscaling (KEDA)** + load tests | Proves the distributed claim with data | P1 |
| 14 | **Terraform IaC + Helm + GitOps (Argo CD)** | Complete cloud story, reproducible deployments | P1/P2 |
| 15 | **Number-level provenance in reports** (click a figure → source SQL/doc) | Trust feature rarely seen | P1 |
| 16 | **Long-term memory** per user/tenant | Personalization and continuity | P1 |
| 17 | **MCP server/client support** | Aligned with emerging standard | P2 |
| 18 | **Reusable playbooks / workflow builder** | Product thinking beyond a chatbot | P2 |
| 19 | **Chaos & resilience tests** (kill workers, broker outage, LLM timeout) | Demonstrates production-readiness | P2 |
| 20 | **Public benchmark report + architecture decision records (ADRs)** | Documentation quality signals senior thinking | P1 |

---

## 5. Data Model (PostgreSQL — core entities)

| Table | Key fields |
|---|---|
| `tenants` | id, name, plan, settings(jsonb), created_at |
| `users` | id, tenant_id, email, password_hash, role, status |
| `api_keys` | id, tenant_id, hashed_key, scopes, expires_at |
| `collections` | id, tenant_id, name, chunking_config, embedding_model, version |
| `documents` | id, collection_id, s3_uri, content_hash, status, metadata(jsonb), version |
| `chunks` | id, document_id, qdrant_point_id, page, section_path, parent_id, text_hash |
| `data_sources` | id, tenant_id, type, encrypted_conn, schema_snapshot, glossary |
| `semantic_layer` | id, data_source_id, table, column, description, embedding_ref |
| `sql_examples` | id, data_source_id, question, sql, embedding_ref, approved_by |
| `threads` | id, tenant_id, user_id, title, summary |
| `runs` | id, thread_id, goal, status, plan(jsonb), budget(jsonb), cost, tokens, started_at, ended_at |
| `run_steps` | id, run_id, node, agent, tool, input, output, latency_ms, tokens, cost, status |
| `checkpoints` | LangGraph Postgres checkpointer tables |
| `approvals` | id, run_id, action, payload, status, decided_by, decided_at |
| `reports` | id, run_id, template_id, version, s3_uri, format |
| `report_templates` | id, tenant_id, name, jinja_body, branding |
| `schedules` | id, tenant_id, cron, workflow, delivery(jsonb), enabled |
| `prompts` | id, name, version, body, status, metrics |
| `feedback` | id, run_id, rating, comment, tags |
| `usage_events` | id, tenant_id, user_id, model, tokens_in, tokens_out, cost, ts |
| `audit_log` | id, tenant_id, actor, action, resource, metadata, ts (append-only) |
| `eval_runs` / `eval_results` | id, dataset, config, metrics(jsonb), commit_sha |

**Qdrant collections:** one collection per embedding model/version, with payload: `tenant_id`, `collection_id`, `document_id`, `page`, `section`, `acl_tags`, `doc_type`, `date`; named dense + sparse vectors. Separate collections for `semantic_layer`, `sql_examples`, and `long_term_memory`.

**Redis keys:** `cache:exact:*`, `cache:sem:*`, `ratelimit:*`, `session:*`, `run:events:*` (pub/sub), `web:cache:*`.

---

## 6. API Surface (v1)

| Area | Endpoints (representative) |
|---|---|
| Auth | `POST /auth/login`, `/auth/refresh`, `/auth/register`, `/api-keys` |
| Documents | `POST /collections`, `POST /collections/{id}/documents`, `GET /documents/{id}/status`, `DELETE /documents/{id}` |
| Query | `POST /query` (RAG, sync/stream), `POST /search` (raw retrieval + scores) |
| Runs | `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/events` (SSE), `POST /runs/{id}/cancel`, `POST /runs/{id}/resume`, `POST /runs/{id}/fork` |
| Approvals | `GET /approvals`, `POST /approvals/{id}/decision` |
| Data | `POST /data-sources`, `GET /data-sources/{id}/schema`, `POST /sql/ask`, `POST /datasets`, `POST /datasets/{id}/profile` |
| Reports | `POST /reports`, `GET /reports/{id}`, `GET /reports/{id}/download?format=pdf`, `POST /schedules` |
| Eval | `POST /eval/runs`, `GET /eval/runs/{id}`, `POST /feedback` |
| Admin | `/admin/users`, `/admin/tenants`, `/admin/usage`, `/admin/cache/flush`, `/admin/prompts`, `/admin/models` |
| System | `/healthz`, `/readyz`, `/metrics` |

All endpoints: typed request/response models, pagination, filtering, consistent error schema, and OpenAPI examples.

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | See success metrics; reranker and embedding inference batched; async I/O throughout |
| **Scalability** | Stateless services; horizontal scaling via HPA; workers scale on queue depth; Qdrant sharding/replication configured for production |
| **Reliability** | Retries, circuit breakers, DLQs, graceful shutdown, zero-downtime rolling deploys, PodDisruptionBudgets, readiness/liveness probes |
| **Availability** | Multi-AZ deployment on AWS; RDS Multi-AZ; Redis replication |
| **Security** | Per Section 3.9; secrets management; encryption at rest (RDS, S3, EBS) and in transit |
| **Maintainability** | Clean layered architecture, dependency injection, ≥ 80% unit-test coverage on core logic, type hints + mypy, ruff/black, pre-commit hooks |
| **Portability** | Runs fully locally via Docker Compose (with Ollama + MinIO) and in the cloud via Helm |
| **Cost control** | Budgets per run/tenant, semantic cache, model router, compression; cost dashboards |
| **Compliance readiness** | Audit logs, data retention controls, PII handling |
| **Documentation** | README, architecture docs, ADRs, runbooks, API docs, demo scripts |

---

## 8. Infrastructure & DevOps

### 8.1 Containerization
- Multi-stage, non-root, minimal-base Dockerfiles per service; pinned dependencies (uv/pip-tools/poetry lock); image caching; health checks.
- `docker-compose.yml` for local: gateway, orchestrator, workers, ingestion, Qdrant, PostgreSQL, Redis, RabbitMQ, MinIO, Ollama, Prometheus, Grafana, OTel Collector, Tempo/Jaeger, Loki.

### 8.2 Kubernetes (EKS)
- Helm chart (or Kustomize) with per-environment values (dev/staging/prod).
- Deployments, Services, Ingress (ALB/NGINX) with TLS (cert-manager), ConfigMaps, Secrets (External Secrets Operator).
- HPA (CPU/RPS) and **KEDA** (RabbitMQ queue length); resource requests/limits; PodDisruptionBudgets; topology spread.
- NetworkPolicies, PodSecurity standards, dedicated namespace and service accounts (IRSA).
- Stateful components: managed services preferred (RDS, ElastiCache, Amazon MQ, S3); Qdrant via Helm/StatefulSet or Qdrant Cloud.
- GPU node group (optional) for reranker/embedding/LLM serving, or CPU-optimized with ONNX/quantization.

### 8.3 AWS
- Terraform modules: VPC, subnets, EKS, ECR, RDS PostgreSQL, ElastiCache Redis, Amazon MQ (RabbitMQ), S3, IAM roles, Secrets Manager, CloudWatch, Route53, ACM, WAF (P2).
- Remote state (S3 + DynamoDB lock); separate dev/staging/prod workspaces.
- Cost guardrails: budgets and alarms, spot nodes for workers (P2).

### 8.4 CI/CD (GitHub Actions)
**On every PR:**
1. Lint (ruff), format check, type check (mypy)
2. Unit tests + coverage report
3. Integration tests (testcontainers: Postgres, Redis, RabbitMQ, Qdrant)
4. Security scans: Trivy (images + IaC), pip-audit, secret scan (gitleaks), Bandit
5. Build Docker images (cached)
6. **Evaluation gate**: run golden-set eval; fail on regression
7. Terraform `fmt/validate/plan` and Helm lint/template

**On merge to main:** build & push to ECR (tagged by SHA, signed) → deploy to staging (Helm/Argo CD) → smoke tests + k6/Locust load test → manual approval → production with **rolling/canary deployment** and automatic rollback on SLO breach.

### 8.5 MLOps Lifecycle
- Versioned artifacts: embedding model, reranker, prompts, chunking configs, and index versions tracked (MLflow or DB registry).
- Reproducible re-indexing job when embedding model/config changes, with blue/green collection swap (alias switch) and zero downtime.
- Offline eval → staging shadow traffic → production rollout; continuous monitoring of quality, drift (query distribution, retrieval score distribution), and cost.
- Feedback loop: logged failures → eval set → prompt/retrieval improvements.

---

## 9. Testing Strategy
| Level | Scope |
|---|---|
| Unit | Chunkers, RRF fusion, cache key logic, SQL validator, guardrails, token budgeter, router rules |
| Integration | Ingestion → Qdrant; retrieval → rerank; agent tool calls; queue retry/DLQ; checkpoint resume |
| Contract | API schemas (schemathesis/OpenAPI), inter-service message schemas |
| Agent/LLM | Deterministic tests with mocked/recorded LLM responses (VCR-style); golden trajectories |
| End-to-end | Scripted scenarios through UI/API (upload → ask → report) |
| Security | Prompt-injection corpus, SQL-injection/forbidden-statement corpus, sandbox escape attempts, tenant isolation tests, RBAC matrix tests |
| Performance | Locust/k6 load, soak, and spike tests; retrieval latency benchmarks |
| Resilience | Kill worker mid-run, broker/DB outage, LLM provider timeout, Qdrant slowdown |

---

## 10. Suggested Repository Structure
```
ai-ops-platform/
├── context.md                # concise, current project context for the AI agent
├── prd.md                    # this document (source of truth for scope)
├── project_notes.md          # running decision/progress log written by the agent
├── README.md
├── docs/                     # architecture, ADRs, runbooks, benchmark report, demo script
├── services/
│   ├── gateway/
│   ├── orchestrator/
│   ├── workers/
│   ├── ingestion/
│   ├── scheduler/
│   ├── sandbox/
│   └── eval/
├── libs/
│   ├── common/               # config, logging, otel, errors, auth utils
│   ├── llm/                  # provider layer, router, prompt registry
│   ├── retrieval/            # hybrid search, rerank, cache, compression
│   ├── agents/               # doc_rag, web, sql, data, report subgraphs
│   ├── guardrails/
│   └── tools/
├── frontend/
├── infra/
│   ├── terraform/
│   ├── helm/
│   └── docker/
├── observability/            # prometheus rules, grafana dashboards (JSON), otel config
├── evals/                    # golden datasets, metrics configs, reports
├── tests/                    # unit, integration, e2e, load, security
├── scripts/
├── docker-compose.yml
├── Makefile
└── .github/workflows/
```

---

## 11. Delivery Roadmap

| Phase | Focus | Deliverables | Exit Criteria |
|---|---|---|---|
| **0. Foundation** | Repo, tooling, config, local stack | Monorepo, Docker Compose, CI skeleton, logging/OTel base, Postgres/Redis/Qdrant/RabbitMQ up | `make up` runs everything; CI green |
| **1. Core RAG** | Ingestion + hybrid retrieval + rerank | Upload→index pipeline, hybrid search, reranker, citations, basic API | Recall@10 baseline measured |
| **2. Optimization** | Semantic cache + compression + CRAG | Cache layer, compressor, token budgeter, adaptive retrieval | Cache hit and compression metrics visible |
| **3. Orchestration** | LangGraph supervisor + checkpoints | Planner/executor/critic, run lifecycle, SSE events, HITL | Multi-step run resumes after crash |
| **4. Specialist Agents** | Web, SQL, Data, Report agents | All five agents integrated as subgraphs; sandbox | End-to-end "question → data → report" works |
| **5. Async & Scale** | RabbitMQ workers, retries, DLQ | Queue-based execution, progress streaming | Load test passes; DLQ flow verified |
| **6. Security & Guardrails** | AuthZ, tenancy, guardrails, audit | RBAC, tenant isolation, injection defenses | Security test corpus passes |
| **7. Observability** | Metrics, traces, dashboards, alerts | Grafana dashboards-as-code, alert rules, cost tracking | Full trace visible for a run |
| **8. Evaluation** | Eval harness + CI gate | Golden sets, RAGAS, SQL eval, regression gate, benchmark report | CI blocks a deliberate regression |
| **9. Cloud & MLOps** | AWS + K8s + Terraform + CD | EKS deploy, KEDA, canary, rollback, blue/green re-index | Public staging URL with SLO dashboard |
| **10. Frontend & Polish** | UI, Run Inspector, admin, demos | Full UI, demo video, README, ADRs | Demo script runs flawlessly |
| **11. Stretch** | MCP, playbooks, Slack bot, chaos tests | Per priority | Optional |

---

## 12. Acceptance Criteria (Definition of Done for v1)
1. A user can upload documents, ask a question, and receive a **cited, grounded** answer with confidence score.
2. A single natural-language goal such as *"Compare last quarter's revenue by region from the database, check the uploaded policy doc for the target, research competitor benchmarks online, and generate a PDF report"* is executed **autonomously** across all five agents, with a visible plan, step timeline, and a downloadable PDF with traceable figures.
3. Risky actions pause for human approval and resume correctly.
4. Killing any worker or service mid-run does not lose the run; it resumes from the last checkpoint.
5. The semantic cache demonstrably reduces latency and cost, with metrics on Grafana.
6. Retrieval/generation/SQL metrics are computed automatically and a regression fails CI.
7. The system is deployed on AWS EKS via Terraform + Helm + CI/CD, with dashboards, alerts, and traces working.
8. Tenant isolation, RBAC, SQL safety, sandbox isolation, and prompt-injection tests pass.
9. Documentation (README, architecture, ADRs, runbooks, benchmark report) and a 3–5 minute demo are complete.

---

## 13. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Scope too large | Strict P0→P1→P2 ordering; each phase has exit criteria; ship vertical slices |
| LLM cost/latency | Model router, caching, compression, local Ollama fallback, budgets |
| Hallucination / wrong SQL | Groundedness checks, SQL validation + dry run, critic loop, HITL |
| Prompt injection via documents/web | Untrusted-content isolation, guardrails, tool allowlists, approvals |
| Cloud cost overrun | Small node groups, spot instances, auto-teardown scripts, AWS Budgets alarms |
| Retrieval quality plateau | Eval-driven iteration, ablations, optional embedding/reranker fine-tuning |
| Distributed-systems complexity | Start with docker-compose; add K8s once flows are stable; ADRs for each decision |
| Vendor lock-in | LiteLLM abstraction, S3/MinIO compatibility, Helm portability |

---

## 14. Open Decisions (to be recorded as ADRs in `project_notes.md`)
- Celery vs. aio-pika consumers for worker runtime.
- Langfuse (self-hosted) vs. custom LLM tracing tables.
- Managed Amazon MQ vs. self-hosted RabbitMQ on K8s.
- Qdrant self-hosted on EKS vs. Qdrant Cloud.
- Frontend: Next.js vs. Streamlit for MVP.
- Sandbox technology: K8s Jobs with gVisor vs. a dedicated sandbox service.
- Cloud LLM provider(s) for production vs. local-only mode for demos.