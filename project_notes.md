# Project Notes & Engineering Log

This log is an ongoing technical learning record and architecture journal for Sahil. Every entry documents what was done, the architectural rationale, alternatives evaluated, trade-offs and risks, deep-dive explanations of non-trivial concepts with code/diagram examples, verification steps, and formal Architecture Decision Records (ADRs).

---

## [2026-10-05] Step 0 — Project Kickoff, PRD Analysis, and Architectural Foundations

### (a) What was done
1. **Full PRD Ingestion & Analysis**: Deep review of `prd.md` across all 14 sections, covering functional requirements (FR-GW, FR-OR, FR-RAG, FR-WEB, FR-SQL, FR-DP, FR-REP, FR-ASYNC, FR-SEC, FR-OBS, FR-EVAL, FR-MOD, FR-UI), non-functional requirements, data models, and the 12-phase roadmap.
2. **Operational Framework Established**:
   - Initialized `context.md` (under 150 lines) as active working memory detailing current state, tech stack, conventions, roadmap checklists, and gaps.
   - Initialized `project_notes.md` as the educational log and ADR register.
3. **P0 Pillar Formalization**: Codified three non-negotiable architectural mandates:
   - **MCP-First Tool Plane**: Decouple tool implementations into standalone MCP servers. Agents invoke tools strictly via MCP protocol clients over JSON-RPC.
   - **Risk-Tiered HITL Governance**: Hard policy gates using LangGraph persisted interrupts backed by HMAC-SHA256 signed single-use approval tokens bound to tool name and argument hashes.
   - **Evaluation Regression Harness**: Baseline quality tracking with Ragas and DeepEval integrated from early slices and enforced in CI/CD.
4. **Resolved Open Decisions**: Evaluated the 7 open decisions from PRD Section 14, establishing 5 foundational ADRs (ADR-001 through ADR-005) and isolating the 2 blocking user questions (primary cloud LLM provider and frontend choice).

---

### (b) Why we chose this approach
A distributed multi-agent platform cannot be built as a monolithic script or a collection of ad-hoc LangChain chains. It requires:
1. **Strict separation of concerns**: Agent reasoning graphs (LangGraph) must not import database drivers, web scrapers, or execution engines directly. If an agent imported database drivers directly, any compromise of agent prompt logic could directly execute arbitrary SQL or access raw connection strings. By mandating an **MCP tool plane**, tools run as isolated microservices with their own authorization, input sanitization, and rate limiting.
2. **Fault-tolerant human-in-the-loop**: High-risk actions (e.g. database mutations, sending external emails, running code against production data) must pause execution safely. If state is stored in memory, a container restart kills the run. By combining PostgreSQL checkpointing with HMAC-signed approval tokens, a paused workflow can survive service crashes, pod rescheduling, or days of human review latency without risk of argument tampering.
3. **Rigorous measurement over vibes**: LLM outputs drift across model versions and prompt edits. Integrating Ragas and DeepEval early guarantees that regressions in retrieval recall, answer faithfulness, or trajectory efficiency are caught in CI before merging.

---

### (c) Alternatives considered and why rejected
- **Direct in-process tool imports (`@tool` decorators in agent code)**:
  - *Why rejected*: Binds agent runtimes directly to infrastructure drivers (SQLAlchemy, browser drivers, subprocesses); makes independent scaling, canary deployments, and per-tool RBAC impossible; increases attack surface for prompt injection and privilege escalation.
- **In-memory approval states or simple database boolean flags (`is_approved = True`)**:
  - *Why rejected*: A boolean flag in a database is vulnerable to race conditions and argument tampering. If an agent pauses to execute `DROP TABLE users`, an attacker or bug could alter the pending arguments to `DROP TABLE audit_log` before approval. Cryptographic token binding guarantees that only the exact approved payload can be executed.
- **Deferring evaluation until Phase 8**:
  - *Why rejected*: Building RAG and agents without evaluation creates technical debt and unquantifiable quality. Without a golden baseline from Phase 1, you cannot know if reranking or hybrid search actually improved retrieval.

---

### (d) Trade-offs and risks
- **Latency overhead of MCP IPC**: Moving tool calls from in-memory function calls to HTTP/SSE JSON-RPC adds ~5–15ms per tool invocation. *Mitigation*: Persistent connection pooling, local network routing in Kubernetes/Docker networks, and asynchronous batching.
- **State persistence overhead**: Writing LangGraph state checkpoints to PostgreSQL on every graph step introduces database writes. *Mitigation*: Async connection pooling, efficient JSONB state representations, and Redis caching for transient step data.
- **Local environment resource footprint**: Running PostgreSQL, Redis, RabbitMQ, Qdrant, MinIO, OTel Collector, Prometheus, and Grafana locally requires ~4–6 GB RAM. *Mitigation*: Minimal local container configurations and resource limits in `docker-compose.yml`.

---

### (e) Non-trivial concepts explained simply

#### 1. The Model Context Protocol (MCP) Tool Plane
In standard agent frameworks, an agent imports Python functions directly:
```python
# Monolithic approach (Tight coupling, security risk)
from my_app.database import execute_sql_query  # Agent process has direct DB access
```
In an **MCP-first architecture**, tools are separate network services speaking the Model Context Protocol (JSON-RPC 2.0). The agent only has an MCP client:
```
[LangGraph Agent] ──(JSON-RPC over HTTP/SSE)──> [MCP Server: SQL Analytics] ──> [PostgreSQL]
```
The agent sends:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "run_read_query",
    "arguments": {"query": "SELECT count(*) FROM orders WHERE status = 'pending';"}
  },
  "id": "req-101"
}
```
**Why this matters**:
- The MCP server independently validates tenant permissions (`tenant_id`), runs AST analysis with `sqlglot` to verify it is strictly a read-only `SELECT`, and enforces row limits.
- The agent runtime never possesses the database credentials.

#### 2. Cryptographically Bound Approval Tokens (HITL Governance)
When an agent reaches a high-risk tool (e.g. `execute_sql_mutation` or `trigger_external_webhook`), it must not execute immediately.
1. LangGraph triggers `interrupt()`: graph execution stops, and the current state is serialized into PostgreSQL.
2. The platform generates an **approval request** containing:
   - `action`: `"execute_sql_mutation"`
   - `payload`: `{"sql": "UPDATE subscriptions SET status = 'active' WHERE id = 42;"}`
   - `run_id`: `"run-uuid-889"`
3. A cryptographic HMAC-SHA256 signature is calculated over these exact parameters:
   $$\text{Token} = \text{HMAC}_{\text{secret}}(\text{run\_id} \,\|\, \text{action} \,\|\, \text{SHA256}(\text{payload}) \,\|\, \text{expires\_at})$$
4. When the human approves via UI/API, the signed approval token is provided to resume the run.
5. The target MCP server verifies the token's signature, checks that the payload hash matches the requested arguments identically, and marks the token as used (single-use replay prevention).
If someone alters the SQL statement after approval was granted, the hash check fails and execution is blocked.

#### 3. Automated Evaluation Gates (Ragas & DeepEval)
Instead of manually guessing whether retrieval works, we compute statistical metrics against a golden dataset:
- **Faithfulness**: Measures whether every claim in the generated answer is directly entailed by the retrieved context (detects hallucinations).
- **Answer Relevancy**: Measures how directly the generated answer addresses the original query.
- **Retrieval Recall@K**: Measures whether the ground-truth relevant document chunk appeared in the top $K$ retrieved chunks.
In CI/CD, if a pull request drops Faithfulness below 0.85 or Recall@10 below 0.90, the pipeline fails and blocks merging.

---

### (f) How to verify it works
1. Verify `context.md` exists, is under 150 lines, and contains the distilled stack, conventions, and phase checklist.
2. Verify `project_notes.md` contains this kickoff entry and the foundational ADRs below.
3. Verify that the open decisions are categorized into adopted ADRs and explicit blocking questions.

---

## Architecture Decision Records (ADRs)

### ADR-001: Worker Runtime — Native Asyncio via aio-pika over Celery
- **Context**: PRD Section 2.3 and Section 14 present an open decision between Celery and `aio-pika` for async task execution and queue consumers.
- **Decision**: Adopt `aio-pika` (or `FastStream` with aio-pika) with native Python `asyncio` consumers rather than Celery.
- **Consequences**:
  - *Positive*: Native integration with FastAPI, LangGraph async subgraphs, and `asyncpg`/SQLAlchemy 2.0 async drivers without thread-blocking issues; granular control over AMQP manual acknowledgements (`ack`/`nack`), prefetch QoS, dead-letter exchanges (DLX), and message header propagation (`X-Correlation-ID`).
  - *Trade-off*: Requires writing custom consumer lifecycle and retry abstractions instead of relying on Celery's built-in decorators, but eliminates Celery's heavy sync worker baggage.

### ADR-002: LLM Observability & Tracing — Hybrid OpenTelemetry Spans + Self-Hosted Langfuse
- **Context**: PRD Section 14 presents an open decision between self-hosted Langfuse vs custom LLM tracing tables.
- **Decision**: Instrument all services with OpenTelemetry for distributed trace propagation across HTTP, RabbitMQ, and MCP servers. For LLM-specific observability (prompt version tracking, token/cost attribution, generation spans), deploy self-hosted Langfuse via Docker Compose / Helm, complemented by our internal append-only PostgreSQL `run_steps` and `audit_log` tables.
- **Consequences**:
  - *Positive*: Provides an enterprise-grade UI for visual trace trees, token breakdown, and cost accounting without SaaS dependencies; maintains internal immutable audit logging in Postgres for security compliance.
  - *Trade-off*: Adds Langfuse service containers to local and cluster deployment profiles.

### ADR-003: Message Broker Deployment — Docker Compose Locally & RabbitMQ Cluster Operator on Kubernetes
- **Context**: PRD Section 14 presents an open decision between Managed Amazon MQ vs self-hosted RabbitMQ on Kubernetes.
- **Decision**: Use the official RabbitMQ container locally in Docker Compose. On Kubernetes (EKS), deploy RabbitMQ using the official RabbitMQ Cluster Operator. Provide a modular Terraform variable to switch to Amazon MQ (RabbitMQ) for cloud production environments that mandate managed infrastructure.
- **Consequences**:
  - *Positive*: Zero cloud dependencies and identical AMQP protocol behavior during local development and testing; automated clustering and quorum queues on K8s; clear production migration path.
  - *Trade-off*: Self-hosted RabbitMQ on K8s requires monitoring disk persistence and operator health.

### ADR-004: Vector Database Deployment — Self-Hosted Qdrant Locally and on EKS
- **Context**: PRD Section 14 presents an open decision between self-hosted Qdrant vs Qdrant Cloud.
- **Decision**: Deploy self-hosted Qdrant via official Docker image locally and Helm chart / StatefulSet on Kubernetes. Support Qdrant Cloud via environment variable configuration (`QDRANT_URL`, `QDRANT_API_KEY`) for deployments opting for managed storage.
- **Consequences**:
  - *Positive*: Complete local offline development without internet connection or external API keys; zero operational costs; strict data sovereignty.
  - *Trade-off*: Production self-hosted Qdrant requires managing persistent volume claims (PVCs) and snapshot backups.

### ADR-005: Code Execution Sandbox — Dedicated Isolated Microservice
- **Context**: PRD Section 14 presents an open decision between K8s Jobs with gVisor vs a dedicated sandbox microservice.
- **Decision**: Build a dedicated `sandbox` microservice exposing an internal HTTP API. In local development and Docker Compose, the sandbox executes Python code inside an isolated container with no network access (`network_mode: none`), dropped Linux capabilities, CPU/memory limits, and ephemeral scratch volumes. In Kubernetes, the sandbox delegates execution to ephemeral K8s Jobs or gVisor (`runsc`) runtime containers with strict `securityContext`.
- **Consequences**:
  - *Positive*: Clean abstraction for the Data Processing Agent (`sandbox_client.execute(code)`); isolates unsafe code execution away from worker processes; works consistently in both Docker Compose and Kubernetes.
  - *Trade-off*: Local sandbox in Docker requires Docker-in-Docker or a sibling container manager pattern.

### ADR-006: Primary LLM Provider & Routing — LiteLLM Abstraction with OpenAI & Local Ollama Fallback
- **Context**: PRD Section 14 lists cloud LLM provider selection for development, evaluation, and production.
- **Decision**: Standardize on LiteLLM as the provider-agnostic interface with OpenAI (`gpt-4o` as primary reasoner/planner, `gpt-4o-mini` for lightweight rewriting/classification) and local Ollama (`llama3.2` / `mistral`) as offline/local fallback.
- **Consequences**:
  - *Positive*: Best-in-class tool calling, structured outputs, and evaluation compatibility; seamless fallback to Ollama when offline or avoiding API charges; vendor-agnostic architecture easily switchable via environment variables.
  - *Trade-off*: Requires valid OpenAI API key for full capability testing, but local Ollama ensures zero blocker when working offline.

### ADR-007: Web Frontend Architecture — Next.js 14+ (App Router), React, and Tailwind CSS
- **Context**: PRD Section 14 and Section 3.13 present an open decision between Next.js vs Streamlit for the user interface.
- **Decision**: Build the production frontend in Next.js (App Router, React, Tailwind CSS, TypeScript).
- **Consequences**:
  - *Positive*: First-class support for real-time SSE event streaming, interactive agent execution step timelines, citation click-throughs, DAG trace visualization (Run Inspector), and responsive multi-tenant dashboards.
  - *Trade-off*: Requires maintaining a TypeScript/React frontend alongside the Python backend, but delivers a true enterprise-grade product rather than a prototype.

---

## [2026-10-05] Phase 0 Completion — Foundation, Tooling, Core Library, Gateway & Local Stack

### (a) What was done
1. **Monorepo Structure Scaffolding**: Established clean layered monorepo layout:
   - `services/`: `gateway` (and stubs for `orchestrator`, `workers`, `ingestion`, `sandbox`, `scheduler`, `eval`).
   - `libs/`: `common` (and packages for `llm`, `retrieval`, `agents`, `guardrails`, `tools`, `mcp_client`).
   - `infra/`: `docker`, `helm`, `terraform`.
   - `observability/`: `otel`, `prometheus`, `grafana`.
   - `tests/`: `unit`, `integration`, `e2e`.
   - `scripts/`: developer tooling and automation.
2. **Packaging & Strict Tooling Configuration**:
   - `pyproject.toml` with pinned Python `>=3.11` dependencies.
   - `ruff` configured with strict lint rules (`E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `ARG`, `SIM`) and automated code formatting.
   - `mypy` configured with `strict = true`, explicit package bases, and zero untyped definitions allowed.
3. **Core Shared Library (`libs/common`)**:
   - [`libs/common/config.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/config.py): Pydantic v2 `BaseSettings` loading typed environment variables from `.env` with sensible local defaults and property flags (`is_production`, `is_testing`).
   - [`libs/common/errors.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/errors.py): Domain exception hierarchy (`AppError`, `NotFoundError`, `ValidationError`, `AuthenticationError`, `AuthorizationError`, `GovernanceError`, `RateLimitError`, `ConflictError`, `ExternalServiceError`) with RFC 7807 problem details dict serialization.
   - [`libs/common/logging.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/logging.py): High-performance structured JSON logging (`structlog`) with automatic contextvar injection (`correlation_id`, `tenant_id`, `run_id`) and active OpenTelemetry span context enrichment (`trace_id`, `span_id`).
   - [`libs/common/telemetry.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/telemetry.py): OpenTelemetry tracer setup with OTLP gRPC export, W3C TraceContext injector/extractor (`traceparent`) for RabbitMQ and MCP protocols, and Starlette/FastAPI `CorrelationIdMiddleware`.
   - [`libs/common/health.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/health.py): `HealthCheckRegistry` supporting `/healthz` (liveness), `/readyz` (readiness with async dependency checks and 503 fallback), and `/metrics` (Prometheus RED metrics exposition).
4. **API Gateway Foundation (`services/gateway`)**:
   - Initialized FastAPI gateway application factory (`create_gateway_app`).
   - Registered `CorrelationIdMiddleware`, CORS middleware, and global exception handlers converting `AppError` and Pydantic `RequestValidationError` into standard RFC 7807 problem details.
   - Exposed system endpoints: `/healthz`, `/readyz`, `/metrics`, and `/api/v1/status`.
5. **Local Infrastructure Topology (`docker-compose.yml`)**:
   - Configured all 9 core local services with container healthchecks: PostgreSQL 16, Redis 7, RabbitMQ 3.13-management, Qdrant 1.11, MinIO S3 storage, OpenTelemetry Collector (gRPC 4317 & Prometheus 8889), Prometheus 2.54, Grafana 11.2, and API Gateway.
   - Multi-stage minimal non-root Dockerfile for API Gateway ([`services/gateway/Dockerfile`](file:///d:/Projects/AI-Operations-Platform/services/gateway/Dockerfile)).
6. **Testing, Automation & CI/CD**:
   - Created 27 unit tests across configuration, errors, logging, telemetry, health registry, and gateway endpoints.
   - Achieved 96% test coverage across `libs` and `services`.
   - Created GitHub Actions CI pipeline (`.github/workflows/ci.yml`), `Makefile`, and PowerShell helper (`scripts/dev.ps1`).

---

### (b) Why we chose this approach
- **Unified Foundation First**: If each agent or microservice invents its own logging, error schemas, or config loading, distributed debugging becomes a nightmare. Having `libs/common` from Phase 0 guarantees that every service, worker, and MCP server created in subsequent phases communicates using identical correlation IDs, RFC 7807 problem details, and telemetry headers.
- **Fail-Fast Typing**: Running `mypy --strict` from the start prevents insidious runtime bugs (like `None` propagation or missing dictionary keys) from entering multi-agent state reducers.
- **Resilient Telemetry**: By making the OTel remote exporter optional and lazy-loading it, tests run in <1 second with zero background network timeouts, while docker-compose runs seamlessly export full gRPC traces to the OTel Collector.

---

### (c) Alternatives considered and why rejected
- **Using Python's standard `logging` without `structlog`**:
  - *Why rejected*: Standard logging produces plain text strings. In a multi-agent system where multiple agents run concurrently, plain text logs interleave unpredictably and cannot be easily filtered by `run_id` or `tenant_id` in Grafana Loki or CloudWatch. `structlog` emits structured JSON where every field is a filterable key.
- **Generic FastAPI `HTTPException`**:
  - *Why rejected*: `raise HTTPException(status_code=400, detail="bad request")` only passes an unstructured string. When an agent or frontend encounters an error, it needs machine-readable error codes (e.g. `GOVERNANCE_BLOCKED` or `RATE_LIMIT_EXCEEDED`) and structured `invalid_params` to trigger automated retry or self-correction logic.
- **Mono-process architecture for Phase 0**:
  - *Why rejected*: While tempting to write a single monolithic script, the PRD mandates microservices and independent MCP servers. Laying out the monorepo structure with distinct `services/` and `libs/` packages ensures that architectural boundaries are preserved from day one.

---

### (d) Trade-offs and risks
- **Strict Typing Friction**: Requiring full type hints and `mypy` strict mode adds a slight amount of authoring time per file. *Benefit*: Eliminates an entire class of production bugs in complex LangGraph state handling.
- **Monorepo Import Paths**: Requiring `libs.` imports across packages requires proper packaging (`pyproject.toml`). *Benefit*: Clean, explicit module resolution with zero circular dependencies.

---

### (e) Non-trivial concepts explained simply

#### 1. RFC 7807 Problem Details
Standard HTTP error responses are often inconsistent:
```json
// Inconsistent API 1:
{"error": "user not found"}

// Inconsistent API 2:
{"message": "Validation failed", "code": 422}
```
RFC 7807 defines a standardized specification for HTTP error responses:
```json
{
  "type": "urn:aiops:error:resource-not-found",
  "title": "RESOURCE_NOT_FOUND",
  "status": 404,
  "detail": "Document not found",
  "error_code": "RESOURCE_NOT_FOUND",
  "invalid_params": {
    "resource_type": "document",
    "resource_id": "doc-123"
  },
  "instance": "/api/v1/collections/col-1/documents/doc-123"
}
```
**Why this is essential for Multi-Agent Systems**:
When an agent (like the SQL Agent or Critic Agent) executes an operation and receives an error, it doesn't need to use regex or prompt an LLM to guess what went wrong. It reads `error_code` and `invalid_params` directly to trigger self-repair loops.

#### 2. W3C Distributed Trace Context Propagation (`traceparent`)
When a request flows through the API Gateway, gets enqueued in RabbitMQ, consumed by an agent worker, and delegated to an MCP server, how does OpenTelemetry connect all these operations into a single continuous trace?
Via the **W3C TraceContext** standard:
$$\text{traceparent: } \texttt{00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01}$$
- `00`: W3C version.
- `4bf92f3577b34da6a3ce929d0e0e4736`: 128-bit **Trace ID** (shared across the entire distributed journey).
- `00f067aa0ba902b7`: 64-bit **Parent Span ID** (identifies the specific calling operation).
- `01`: Trace flags (`01` = sampled/recorded).

Our [`libs/common/telemetry.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/telemetry.py) functions `inject_trace_context(carrier)` and `extract_trace_context(carrier)` inject this header into HTTP headers, AMQP message properties, and MCP request metadata, ensuring end-to-end trace waterfalls in Grafana/Jaeger.

#### 3. Python Contextvars in Concurrent Event Loops
In synchronous multi-threaded applications, thread-local storage (`threading.local()`) is used to store request context (e.g. current user ID).
However, in asynchronous Python (`async`/`await`), multiple concurrent requests run on the **same single OS thread** via the event loop! If you used `threading.local()`, Request B could overwrite Request A's tenant ID!
Python's `contextvars.ContextVar` solves this:
```python
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)
```
Whenever an `async` task is scheduled or context is switched, Python automatically swaps the active contextvar state. Our [`CorrelationIdMiddleware`](file:///d:/Projects/AI-Operations-Platform/libs/common/telemetry.py) sets the correlation ID at request entry, and structlog processors automatically read it on every log call without passing `correlation_id` through every function parameter.

---

### (f) How to verify it works
1. Run all unit tests with coverage:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit -v --cov=libs --cov=services --cov-report=term-missing
   ```
   *Expected output*: 27 passed, >= 96% coverage across all modules.
2. Run Ruff linter and formatter:
   ```bash
   .\.venv\Scripts\ruff.exe check .
   .\.venv\Scripts\ruff.exe format --check .
   ```
   *Expected output*: All checks passed, 20 files already formatted.
3. Run Mypy strict type analysis:
   ```bash
   .\.venv\Scripts\mypy.exe libs services
   ```
   *Expected output*: `Success: no issues found in 10 source files`.
4. Verify Docker Compose stack configuration:
   ```bash
   docker compose config
   ```
   *Expected output*: Valid YAML syntax resolving all 9 services, networks, and volumes.

---

## [2026-10-05] Phase 1 (Slice 1.1) — Layout-Aware Document Parsing & Multi-Strategy Chunking

### (a) What was done
1. **Domain Data Models ([`libs/retrieval/models.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/models.py))**:
   - Defined `DocumentMetadata` capturing mandatory `tenant_id`, `collection_id`, `doc_type`, `acl_tags`, and SHA-256 `content_hash` (FR-RAG-6, FR-RAG-7).
   - Defined `ParsedBlock` and `ParsedDocument` representing layout components (headings with hierarchy levels, tables with column headers, paragraphs, code blocks, page numbers).
   - Defined `Chunk` model with `parent_id`, `is_parent`, `section_path`, `page_number`, `token_count`, and `content_hash`.
2. **Layout-Aware Multi-Format Parser ([`libs/retrieval/parser.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/parser.py))**:
   - Implemented `DocumentParser` supporting Markdown, plain text, HTML (`BeautifulSoup`), CSV, and PDF (`pypdf`).
   - Markdown parser tracks `#` heading breadcrumbs (`section_path`) and detects markdown table grids.
   - HTML parser extracts `<title>`, headings (`h1`–`h6`), tables (`<th>`, `<tr>`, `<td>`), and `<pre>` code.
   - CSV parser converts structured tabular data into formatted markdown table blocks with retained column headers.
   - PDF parser extracts text page-by-page, assigning exact `page_number` provenance for downstream citations.
   - Content hash calculation via SHA-256 on raw bytes for deduplication.
3. **Multi-Strategy Document Chunker ([`libs/retrieval/chunker.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/chunker.py))**:
   - `RECURSIVE`: Recursive character splitting using hierarchical separators (`\n\n`, `\n`, `. `, ` `) with sliding-window overlap.
   - `HEADING_AWARE`: Groups blocks under their parent heading path to prevent crossing major topical section boundaries.
   - `TABLE_AWARE`: Splits large tables while repeating table headers at the top of every chunk slice, ensuring row semantics are never lost.
   - `PARENT_CHILD`: Creates large parent context chunks (`parent_chunk_size`, e.g. 2000 chars) and smaller child chunks (`chunk_size`, e.g. 500 chars) linked via `parent_id` (FR-RAG-5, FR-RAG-16).
4. **Unit Tests & Verification**:
   - Wrote 10 new tests in [`tests/unit/test_parser.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_parser.py) and [`tests/unit/test_chunker.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_chunker.py). Total test count increased from 27 to 37 (100% passing).
   - Strict `mypy` and `ruff` checks passing with 0 warnings.

---

### (b) Why we chose this approach
- **Structure-Preserving Parsing over Blind Extraction**: Standard RAG pipelines treat all documents as unformatted text strings. When tables or headings are flattened, critical relationship data is destroyed (e.g. an LLM cannot tell which column an entry belongs to). Our layout parser retains heading breadcrumbs (`section_path`) and table headers, allowing the retrieval engine to supply structural context to the LLM.
- **Parent-Child Chunking**: Dense vector search works best on small, highly specific chunks (200–500 chars) where semantic signal isn't diluted. However, LLMs generate better answers when given broad surrounding context (1000–2000 chars). By storing parent-child relationships, we embed small child chunks for search accuracy, but return the broader parent chunk during synthesis.

---

### (c) Alternatives considered and why rejected
- **Fixed-character windowing (`text[i:i+500]`)**:
  - *Why rejected*: Blind windowing splits words, sentences, and table rows in half. It produces low retrieval accuracy and disjointed generation.
- **External Unstructured API service**:
  - *Why rejected*: Unstructured API adds external cloud latency, API key costs, and a network failure point. Pure-Python parsing using standard libraries (`pypdf`, `BeautifulSoup`, `csv`, regex) runs locally in milliseconds with zero external dependencies.

---

### (d) Trade-offs and risks
- **Storage expansion in Parent-Child mode**: Storing both parent and child chunks increases text storage volume by ~1.5x. *Mitigation*: Only child chunks are vectorized and indexed in Qdrant; parent chunks are stored as payload attributes or database rows, avoiding vector RAM explosion.

---

### (e) Non-trivial concepts explained simply

#### 1. Parent-Child (Small-to-Big) Chunking
In traditional RAG, there is a fundamental conflict in choosing chunk size:
- **Small chunks (e.g., 200 chars)**: High embedding precision (the vector is focused on a single fact), but poor context for generation (the LLM cannot understand the broader paragraph).
- **Large chunks (e.g., 2000 chars)**: Rich context for generation, but diluted embedding precision (the vector averages many different topics, causing retrieval misses).

**The Solution: Small-to-Big Retrieval**:
```
┌────────────────────────────────────────────────────────┐
│ Parent Chunk (2000 chars) - ID: "p-101"                │
│ ┌───────────────────┐ ┌───────────────────┐            │
│ │ Child Chunk A     │ │ Child Chunk B     │  ...       │
│ │ (Embedded vector) │ │ (Embedded vector) │            │
│ └───────────────────┘ └───────────────────┘            │
└────────────────────────────────────────────────────────┘
```
1. Embed and index only **Child Chunks** in Qdrant.
2. When the user asks a question, Qdrant retrieves Child Chunk A with high precision.
3. Before passing context to the LLM, the retrieval engine resolves `parent_id = "p-101"` and injects the complete **Parent Chunk** into the prompt.

#### 2. Table-Aware Chunking with Header Repetition
Consider a Markdown table of server latencies:
```markdown
| Region | p50 (ms) | p99 (ms) | Cost ($) |
| --- | --- | --- | --- |
| us-east-1 | 12 | 45 | 120 |
... (20 rows later) ...
| ap-south-1 | 18 | 62 | 95 |
```
If a naive chunker cuts this table in half, Chunk 2 contains:
```markdown
| ap-south-1 | 18 | 62 | 95 |
```
An LLM reading Chunk 2 has no idea whether `18` is latency, CPU utilization, or price!
Our [`_chunk_table_aware`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/chunker.py) detects table blocks and prepends the original header row:
```markdown
| Region | p50 (ms) | p99 (ms) | Cost ($) |
| --- | --- | --- | --- |
| ap-south-1 | 18 | 62 | 95 |
```
This guarantees that tabular chunks remain fully interpretable to embedding models and LLMs.

---

### (f) How to verify it works
1. Run parser and chunker unit tests:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_parser.py tests/unit/test_chunker.py -v
   ```
   *Expected output*: 10 passed in <0.2s.
2. Run full unit test suite:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit -v
   ```
   *Expected output*: 37 passed.

---

## [2026-10-06] Phase 1 (Slice 1.2) — Dense & Sparse BM25 Embeddings and Qdrant Hybrid Indexing

### (a) What was done
1. **BM25 Sparse Vector Encoder ([`libs/retrieval/sparse.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/sparse.py))**:
   - Implemented `BM25SparseEncoder` implementing BM25 term weighting:
     $$w_t = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot \left(1 - b + b \cdot \frac{\text{doc\_len}}{\text{avg\_len}}\right)}$$
   - Applied positive 31-bit token hashing trick (`hashlib.md5(token)[:8] % 1_000_000`) and logarithmic weight compression.
   - Outputs sorted `rest.SparseVector(indices=..., values=...)` strictly matching Qdrant's sparse vector index specification.
2. **Dense Vector Embedding Generator ([`libs/retrieval/embeddings.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/embeddings.py))**:
   - Implemented `DenseEmbeddingModel` with batching, L2 vector normalization (unit length for cosine distance), and lazy model loading.
   - Built a deterministic mock embedding mode enabling fast unit test execution (<0.01s per test) with zero external network downloads.
3. **Qdrant Hybrid Indexer & Collection Manager ([`libs/retrieval/indexer.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/indexer.py))**:
   - Configured dual named vector collections: `"dense"` (size 384, cosine distance) and `"sparse"` (sparse vector index).
   - Created keyword payload indices on `tenant_id`, `collection_id`, `document_id`, `is_parent` for pre-retrieval filtering (FR-RAG-13).
   - Implemented batch point upsert with complete provenance metadata (chunk ID, document ID, page number, section breadcrumbs, parent ID, token count, content hash).
   - Implemented cascading document deletion (`delete_document`) scoped by tenant ID (FR-RAG-10).
4. **Unit Tests & Verification**:
   - Created [`tests/unit/test_sparse.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_sparse.py), [`tests/unit/test_embeddings.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_embeddings.py), and [`tests/unit/test_indexer.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_indexer.py). Total unit tests increased from 37 to 46 (100% passing).
   - Strict `mypy` and `ruff` checks passing with 0 errors.

---

### (b) Why we chose this approach
- **Unified Hybrid Storage in Qdrant**: Traditional architectures run an Elasticsearch/OpenSearch cluster for BM25 and a separate vector database for dense embeddings. This doubles operational cost, cloud footprint, and data synchronization headaches. Qdrant natively supports both named dense vectors and sparse inverted vectors in the same collection, allowing atomic upserts and unified queries.
- **Pre-Retrieval Tenant Filtering**: Multi-tenancy must never be an afterthought. By indexing `tenant_id` as a keyword payload field and configuring payload indexes, queries will apply tenant filters at the vector index level, ensuring zero data leakage between organizations.

---

### (c) Alternatives considered and why rejected
- **Dense-only vector search**:
  - *Why rejected*: Pure dense embeddings are semantic; they struggle with exact keyword matches like part numbers ("SKU-4912"), error codes ("ERR-10061"), and proper nouns. Sparse BM25 indexing guarantees that exact keyword queries receive top ranking.
- **Separate BM25 search engine (Elasticsearch)**:
  - *Why rejected*: Significant operational overhead, memory consumption (JVM heap), and dual-write consistency issues. Qdrant provides both dense and sparse retrieval in a single lightweight Rust binary.

---

### (d) Trade-offs and risks
- **Index memory in Qdrant**: Storing two vectors per chunk increases memory usage. *Mitigation*: Sparse vectors only store non-zero term indices and weights (typically 20–50 non-zero elements per chunk), making sparse index overhead a fraction of dense vector memory.

---

### (e) Non-trivial concepts explained simply

#### 1. Dense vs. Sparse Vectors in Hybrid Retrieval
Imagine a user searching for:
$$\text{Query: "Fix for error ERR-404-AUTH in gateway"}$$
- **Dense Vector (Bi-encoder)**:
  Compresses the sentence into 384 floating-point numbers based on general semantic meaning. The dense model knows "Fix" is similar to "Resolve", and "error" is similar to "exception". But it may blur "ERR-404-AUTH" into a generic "error code" embedding, matching unrelated errors like "ERR-500-INTERNAL".
- **Sparse Vector (BM25)**:
  Represents exact keyword tokens as high-dimensional coordinates:
  $$\text{Vector: } \{\text{index}(ERR\text{-}404\text{-}AUTH): 3.25, \; \text{index}(gateway): 1.12\}$$
  Any document mentioning the exact token "ERR-404-AUTH" receives an enormous score boost.

By indexing **both** in Qdrant, our upcoming hybrid retriever merges semantic understanding with exact token precision.

#### 2. Deterministic Hash Trick for Sparse Vocabulary
Traditional BM25 requires maintaining a global dictionary mapping every unique word to an ID (`{"the": 0, "platform": 1, ...}`). In distributed systems, keeping a synchronized global vocabulary across multiple worker processes requires shared state or database locks.
Instead, we use the **Hashing Trick**:
$$\text{Token Index} = \text{MD5}(\text{token}) \pmod{1{,}000{,}000}$$
Every worker or server calculates the identical index for any word instantly, with zero network communication or dictionary synchronization.

---

### (f) How to verify it works
1. Run sparse, embeddings, and indexer unit tests:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_sparse.py tests/unit/test_embeddings.py tests/unit/test_indexer.py -v
   ```
   *Expected output*: 9 passed in <0.3s.
2. Run full test suite with coverage:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit -v --cov=libs --cov=services
   ```
   *Expected output*: 46 passed, 90% coverage.

---

## [2026-10-06] Phase 1 (Slice 1.3) — Hybrid Retrieval, Reciprocal Rank Fusion, Cross-Encoder Reranker & Parent Expansion

### (a) What was done
1. **Reciprocal Rank Fusion (RRF) Implementation ([`libs/retrieval/hybrid.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/hybrid.py))**:
   - Implemented `calculate_rrf_score` using the standard rank fusion formula:
     $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
     with configurable smoothing parameter $k$ (default 60) and modality weights ($w_{\text{dense}}$, $w_{\text{sparse}}$).
2. **Hybrid Retriever with Tenant Filtering ([`libs/retrieval/hybrid.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/hybrid.py))**:
   - `HybridRetriever` performs concurrent queries against Qdrant using `query_points`:
     - Dense search on named vector `"dense"` with cosine distance.
     - Sparse search on named vector `"sparse"` with BM25 term weights.
   - Enforces pre-retrieval filtering (`_build_filter`) with mandatory `tenant_id` and exclusion of container parent chunks (`is_parent=False`).
   - Merges results using RRF and populates `SearchResult` models with dense and sparse rank provenance.
   - Implemented **Parent Context Expansion (FR-RAG-16)**: automatically batch-retrieves parent container chunks from Qdrant by ID and populates `expanded_content`, providing wide contextual windows to downstream LLMs.
3. **Cross-Encoder Reranker & MMR Diversity ([`libs/retrieval/reranker.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/reranker.py))**:
   - Implemented `CrossEncoderReranker` supporting both `sentence_transformers.CrossEncoder` and an ultra-fast deterministic fallback scorer analyzing unigram coverage, bigram overlap, and exact phrase proximity.
   - Score threshold filtering (`min_score`) to drop irrelevant candidate chunks.
   - **Maximal Marginal Relevance (MMR) Re-Selection (FR-RAG-15)**: penalizes candidate chunks that have high token Jaccard redundancy with already selected chunks, ensuring diverse top-$K$ contexts.
4. **Unit Tests & Verification**:
   - Created [`tests/unit/test_hybrid.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_hybrid.py) (5 tests) and [`tests/unit/test_reranker.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_reranker.py) (4 tests).
   - Total test count expanded from 46 to 55 (100% passing).
   - Strict `mypy` and `ruff` checks passing with 0 errors across 34 source files.

---

### (b) Why we chose this approach
- **Reciprocal Rank Fusion over Score Normalization**: Dense cosine similarities (e.g. 0.72) and sparse BM25 scores (e.g. 14.8) operate on completely different numerical scales and distributions. Standard min-max normalization is extremely sensitive to outlier scores and varying document lengths. RRF is scale-invariant because it operates purely on **ranks** (1st, 2nd, 3rd) rather than raw magnitudes.
- **Two-Stage Retrieval (Retrieve-then-Rerank)**: Bi-encoders (dense/sparse) are fast ($O(1)$ indexed lookup) but evaluate query and document independently. Cross-encoders examine the full interaction of query and document tokens ($O(N \cdot M)$ full attention), providing superior ranking accuracy. Reranking top-40 candidates down to top-8 gives the speed of bi-encoders with the precision of cross-encoders.
- **MMR Diversity**: When retrieving from long technical documents, multiple adjacent chunks often contain repetitive boilerplate or similar sentences. MMR prevents the prompt context window from being flooded with redundant duplicates, leaving room for diverse factual evidence.

---

### (c) Alternatives considered and why rejected
- **Direct linear score combination ($\alpha \cdot \text{dense} + (1-\alpha) \cdot \text{sparse}$)**:
  - *Why rejected*: BM25 scores are unbounded and change drastically depending on document collection size, while cosine scores lie between -1 and 1. Direct weighted sums require constant tuning of scale factors per collection. RRF works out-of-the-box across all collections without calibration.
- **Reranking all chunks directly with Cross-Encoder**:
  - *Why rejected*: Cross-encoders cannot be pre-indexed into a vector database because every query-document pair must pass through the transformer together. Running a cross-encoder across thousands of chunks per query would take seconds and destroy system latency.

---

### (d) Trade-offs and risks
- **Latency of Two-Stage Pipeline**: Querying Qdrant twice (dense + sparse) and running a reranker adds additional pipeline stages. *Mitigation*: Qdrant queries execute against the same local/remote cluster in single-digit milliseconds; fallback cross-scoring takes <1ms, and neural rerankers are restricted to top 40 candidates with batched inference.

---

### (e) Non-trivial concepts explained simply

#### 1. Reciprocal Rank Fusion (RRF)
Suppose we search for *"PostgreSQL read replica lag"* and get these ranked lists:
- **Dense Vector Search**:
  1. Doc A (High semantic similarity)
  2. Doc B
  3. Doc C
- **Sparse BM25 Search**:
  1. Doc C (Exact token match for "lag")
  2. Doc D
  3. Doc A

With standard RRF parameter $k = 60$:
$$\text{RRF}(\text{Doc A}) = \frac{1}{60 + 1} + \frac{1}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = \mathbf{0.03226}$$
$$\text{RRF}(\text{Doc C}) = \frac{1}{60 + 3} + \frac{1}{60 + 1} = \frac{1}{63} + \frac{1}{61} \approx 0.01587 + 0.01639 = \mathbf{0.03226}$$
$$\text{RRF}(\text{Doc B}) = \frac{1}{60 + 2} + 0 = \frac{1}{62} \approx \mathbf{0.01613}$$

Notice that Doc A and Doc C, which appear near the top of **both** lists, get roughly double the score of Doc B, which only appeared in one list. Documents supported by multiple retrieval modalities rise to the top!

#### 2. Bi-Encoder vs. Cross-Encoder
- **Bi-Encoder (Embedding Models)**:
  $$\text{Query} \rightarrow \text{Model} \rightarrow \mathbf{v}_q$$
  $$\text{Document} \rightarrow \text{Model} \rightarrow \mathbf{v}_d$$
  $$\text{Score} = \cos(\mathbf{v}_q, \mathbf{v}_d)$$
  The query and document never see each other until the final dot product. Fast, but misses fine-grained token-to-token semantic interactions.
- **Cross-Encoder (Rerankers)**:
  $$[\text{Query}, \text{Document}] \rightarrow \text{Transformer (Full Attention)} \rightarrow \text{Score}$$
  Every token in the query attends directly to every token in the document. It catches subtle negation, modifiers, and exact context matches that bi-encoders miss.

#### 3. Maximal Marginal Relevance (MMR)
When selecting the next chunk $d_i$ to include in the context:
$$\text{MMR}(d_i) = \lambda \cdot \text{Score}(q, d_i) - (1 - \lambda) \cdot \max_{d_j \in \text{Selected}} \text{Sim}(d_i, d_j)$$
If Candidate B is highly relevant, but has 90% word overlap with Candidate A (which is already selected), the redundancy penalty $(1 - \lambda) \cdot 0.9$ drops its score, allowing Candidate C (which covers a new angle of the question) to be selected instead.

---

### (f) How to verify it works
1. Run hybrid retrieval and reranker unit tests:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_hybrid.py tests/unit/test_reranker.py -v
   ```
   *Expected output*: 9 passed in <1.5s.
2. Run full test suite:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit -v
   ```
   *Expected output*: 55 passed.

---

## [2026-10-06] Phase 1 (Slice 1.4) — Citation Engine, Grounded Answer Generation & Refusal Behavior

### (a) What was done
1. **Citation Engine ([`libs/retrieval/citations.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/citations.py))**:
   - Implemented `Citation` model capturing exact chunk ID, document UUID, page number, section breadcrumbs, and passage excerpt (FR-RAG-26).
   - `CitationEngine.format_context`: Structures retrieved chunks into numbered `[Source N]` context blocks for the LLM prompt.
   - `CitationEngine.extract_citations`: Extracts inline bracketed citations (`[1]`, `[2]`, `[1, 2]`) and correlates them back to source provenance objects while detecting hallucinated or unmapped references (`unmapped_citations`).
   - `CitationEngine.is_refusal`: Detects insufficient evidence responses (`"I don't know based on the provided documents"`, etc.) satisfying FR-RAG-28.
   - `CitationEngine.verify_groundedness`: Sentence-by-sentence claim validation evaluating lexical grounding ratios against cited excerpts; flags unsupported statements (`unsupported_claims`) and returns calibrated confidence score (FR-RAG-27).
2. **Provider-Agnostic LLM Layer ([`libs/llm/provider.py`](file:///d:/Projects/AI-Operations-Platform/libs/llm/provider.py))**:
   - Defined `LLMProvider` protocol, `LLMMessage`, and `LLMResponse`.
   - Built `MockLLMProvider` for deterministic testing with zero network overhead.
   - Built `LiteLLMProvider` for multi-provider routing (OpenAI, Anthropic, Bedrock, Ollama) with automated fallback.
3. **End-to-End Grounded Answer Generator ([`libs/retrieval/generator.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/generator.py))**:
   - Implemented `RAGGenerator` orchestrating hybrid retrieval -> cross-encoder rerank -> context assembly -> prompt synthesis -> citation extraction & groundedness verification.
   - Implemented explicit guard: when no candidates exist or all fall below `min_relevance`, returns standard refusal with `is_refusal=True` and confidence 1.0 (FR-RAG-28).
4. **Unit Tests & Verification**:
   - Created [`tests/unit/test_citations.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_citations.py) (4 tests) and [`tests/unit/test_generator.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_generator.py) (2 tests).
   - Total test count expanded from 55 to 61 (100% passing).
   - Mypy strict and Ruff checks pass cleanly across all 39 source files.

---

### (b) Why we chose this approach
- **Strict Grounding over Unconstrained Generation**: Enterprise operations cannot tolerate hallucinations. If an agent answers a question about deployment policy or database replicas with plausible-sounding fiction, systems break. By forcing the LLM into a numbered source format and validating citations sentence-by-sentence, every claim in the generated answer is verifiable.
- **Explicit "I Don't Know" Fallback**: When retrieval fails to find relevant chunks or scores are below minimum thresholds, forcing an LLM to generate an answer inevitably creates hallucinated filler. Short-circuiting directly to a refusal preserves system integrity and informs supervisory agents that alternative research (e.g. web search) is required.

---

### (c) Alternatives considered and why rejected
- **Relying solely on LLM self-attestation for citations**:
  - *Why rejected*: LLMs frequently hallucinate nonexistent citation numbers (`[5]` when only 2 sources were provided). Our `CitationEngine` explicitly parses every `[N]` reference, cross-checks it against the candidate set, and flags unmapped citations.
- **Hardcoding OpenAI API directly**:
  - *Why rejected*: Violates the PRD's provider-agnostic and local Ollama fallback requirements. The `LLMProvider` protocol enables seamless swapping between cloud LLMs and local runtimes with zero code changes.

---

### (d) Trade-offs and risks
- **Sentence-level token overlap for groundedness**: Pure token overlap is computationally cheap (<1ms), but can occasionally miss subtle semantic contradictions. *Mitigation*: In Phase 8, we will augment this with formal DeepEval/Ragas NLI judge models as an offline evaluation gate.

---

### (e) Non-trivial concepts explained simply

#### 1. Inline Citation Mapping & Provenance Graph
When an answer states:
> *"Microservices communicate over gRPC [1] and require signed HMAC tokens for mutations [2]."*

The frontend or user needs to click `[1]` and see:
- Document: `Architecture_Spec.pdf`
- Page: 3
- Section: `Architecture > Observability`
- Exact excerpt text from the original PDF chunk.

Our `CitationEngine` bridges the generated text with the database chunk by numbering candidates `[Source 1]`, `[Source 2]`, parsing the generated brackets, and emitting structured `Citation` objects alongside the text.

#### 2. Hallucination Detection & Claim Grounding
A generated answer consists of individual claims $S_1, S_2, \dots, S_n$.
For each claim $S_i$:
1. Extract the cited source $C_k$.
2. Verify that informative tokens in $S_i$ are present in $C_k$'s excerpt.
3. If token coverage is $< 0.5$, flag $S_i$ in `unsupported_claims` and reduce the answer's `confidence_score`.

---

### (f) How to verify it works
1. Run citation engine and generator tests:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_citations.py tests/unit/test_generator.py -v
   ```
   *Expected output*: 6 passed in <1.5s.
2. Run full test suite:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit -v
   ```
   *Expected output*: 61 passed.

---

## [2026-10-06] Phase 1 (Slice 1.5) — Evaluation Harness Plumbing, Golden Dataset & Baseline Benchmark

### (a) What was done
1. **Golden Evaluation Dataset ([`evals/datasets/golden_rag.json`](file:///d:/Projects/AI-Operations-Platform/evals/datasets/golden_rag.json))**:
   - Authored technical operational domain dataset across PostgreSQL tuning (PgBouncer), Patroni/HA failover, Kubernetes autoscaling (HPA), KEDA event-driven queue workers, HITL governance token rules, tenant isolation policies, OpenTelemetry distributed tracing standards, and Redis two-tier caching.
   - Structured with `documents` (chunks with full section paths and hashes) and `queries` with ground truth `relevant_chunk_ids` and reference answers.
2. **Information Retrieval & Faithfulness Metrics ([`evals/metrics.py`](file:///d:/Projects/AI-Operations-Platform/evals/metrics.py))**:
   - `recall_at_k`: Proportion of ground truth relevant chunks retrieved in the top $K$.
   - `mrr_score` (Mean Reciprocal Rank): Reciprocal of the 1-indexed rank of the first relevant chunk ($1/\text{rank}$).
   - `hit_rate_at_k`: Binary indicator of whether at least one relevant chunk was retrieved in the top $K$.
   - `ndcg_at_k`: Normalized Discounted Cumulative Gain accounting for rank positions.
   - `lexical_faithfulness`: Token overlap ratio of answer statements with reference context.
3. **Automated Evaluation Benchmark Runner ([`evals/runner.py`](file:///d:/Projects/AI-Operations-Platform/evals/runner.py))**:
   - `EvaluationRunner` indexes the golden dataset and runs ablation evaluations across 4 search configurations:
     1. `dense_only`: Bi-encoder dense vectors only.
     2. `sparse_only`: BM25 lexical vectors only.
     3. `hybrid`: Dense + BM25 merged via Reciprocal Rank Fusion ($k=60$).
     4. `hybrid_rerank`: Hybrid RRF + Cross-Encoder reranking.
   - Measures and logs latency, Recall@10, MRR, Hit Rate@10, and NDCG@10.
4. **Baseline Measurement (Phase 1 Exit Criteria Verified)**:
   - Evaluated benchmark on golden technical dataset:
     - `dense_only`: Recall@10 = 1.0, MRR = 0.2369, NDCG@10 = 0.4133, Latency = 1.16ms
     - `sparse_only`: Recall@10 = 1.0, MRR = 1.0, NDCG@10 = 1.0, Latency = 1.88ms
     - `hybrid`: Recall@10 = 1.0, MRR = 1.0, NDCG@10 = 1.0, Latency = 1.12ms
     - `hybrid_rerank`: Recall@10 = 1.0, MRR = 1.0, NDCG@10 = 1.0, Latency = 1.28ms
   - **Exit Criteria Status**: Phase 1 Exit Criteria (*"Recall@10 baseline measured"*) is achieved and recorded.
5. **Unit & Regression Tests ([`tests/unit/test_evals.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_evals.py))**:
   - Added 5 unit tests verifying metric formulas and running the baseline benchmark. Total tests increased to 66 (100% passing).
   - Strict `mypy` and `ruff` passing cleanly across all 43 source files.

---

### (b) Why we chose this approach
- **Eval-Driven Engineering from Phase 1**: Too many AI projects build complex architectures without measuring retrieval quality until late in development. By establishing golden datasets, metric functions, and ablation runners in Phase 1, every optimization in Phase 2 (semantic caching, context compression, token pruning) can be evaluated against this concrete baseline.
- **Ablation Comparison**: Running `dense_only`, `sparse_only`, `hybrid`, and `hybrid_rerank` side-by-side demonstrates the exact value of each component. For exact configuration queries (e.g. "PgBouncer max_client_conn"), sparse BM25 provides instant rank-1 precision, while dense alone ranks it lower (MRR 0.24 vs 1.00), proving why hybrid retrieval is non-negotiable for enterprise operations.

---

### (c) Alternatives considered and why rejected
- **Deferring evaluation until Phase 8**:
  - *Why rejected*: Violates the PRD's P0 core pillar ("Start the evaluation harness early... so quality is measured from the first RAG slice"). Without automated metrics, regression detection is impossible.
- **Sole reliance on cloud-based LLM-as-a-judge for retrieval metrics**:
  - *Why rejected*: LLM judges are non-deterministic, slow, expensive, and cannot be run on every local test run. Ground-truth IR metrics (Recall@K, MRR, NDCG) are deterministic, instantaneous (<2ms), and mathematically rigorous.

---

### (d) Trade-offs and risks
- **Synthetic/Hand-curated Golden Set Scale**: The initial golden dataset contains 8 representative technical documents and 5 evaluation queries. *Mitigation*: In Phase 8, we will expand this dataset to 200+ complex real-world queries covering multi-hop questions, contradictory facts, and edge cases.

---

### (e) Non-trivial concepts explained simply

#### 1. Recall@K vs. Mean Reciprocal Rank (MRR)
- **Recall@K**:
  "Did the retrieval engine find all the needle(s) in the haystack?"
  If a question requires 2 documents to answer, and both appear in the top 10 results, Recall@10 is $100\%$.
- **MRR (Mean Reciprocal Rank)**:
  "How close to the top was the first correct answer?"
  $$\text{MRR} = \frac{1}{\text{Rank of first relevant chunk}}$$
  - If the first correct chunk is at Rank 1: $\text{RR} = 1/1 = \mathbf{1.0}$.
  - If the first correct chunk is at Rank 4: $\text{RR} = 1/4 = \mathbf{0.25}$.
  High MRR is vital because humans and LLMs pay the most attention to top-ranked chunks.

#### 2. NDCG (Normalized Discounted Cumulative Gain)
NDCG penalizes relevant documents that appear further down the ranked list using a logarithmic discount factor $\frac{1}{\log_2(\text{rank} + 1)}$:
- A relevant document at Rank 1 provides full gain ($1.0 / \log_2(2) = 1.0$).
- A relevant document at Rank 10 provides only fractional gain ($1.0 / \log_2(11) \approx 0.29$).
NDCG normalizes this against the ideal ordering (IDCG), producing a score between 0.0 and 1.0 that measures ranking quality.

---

### (f) How to verify it works
1. Run evaluation tests and view the live benchmark output:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_evals.py -v -s
   ```
   *Expected output*: 5 passed, ablation metrics printed for dense, sparse, hybrid, and hybrid_rerank.
2. Run full test suite with coverage:
   ```bash
   .\.venv\Scripts\pytest.exe --cov=libs --cov=evals --cov=services -v
   ```
   *Expected output*: 66 passed, 89% total coverage.

---

## [2026-10-06] Phase 1 (Slice 1.6) — Gateway Ingestion & Search API Endpoints (Phase 1 Complete)

### (a) What was done
1. **Dependency Injection Providers ([`services/gateway/dependencies.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/dependencies.py))**:
   - Built lazy dependency providers: `get_qdrant_client`, `get_dense_model`, `get_sparse_encoder`, `get_indexer`, `get_retriever`, `get_reranker`, `get_citation_engine`, `get_llm_provider`, and `get_rag_generator`.
   - In testing mode (`ENVIRONMENT="testing"`), cleanly serves in-memory Qdrant with zero port conflicts.
2. **Gateway RAG API Router ([`services/gateway/rag_router.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/rag_router.py))**:
   - `POST /api/v1/collections`: Initializes tenant-isolated Qdrant collection with dual named vectors and payload indices.
   - `POST /api/v1/collections/{id}/documents`: Ingests document text/markdown -> layout-aware parsing -> chunking -> batch embedding -> Qdrant upsert; returns document UUID, chunk count, and SHA-256 hash.
   - `DELETE /api/v1/collections/{id}/documents/{doc_id}?tenant_id={tid}`: Cascading deletion of document points under mandatory tenant isolation (FR-RAG-10).
   - `POST /api/v1/search`: Hybrid retrieval with Reciprocal Rank Fusion, parent expansion, and cross-encoder reranking; returns ranked `SearchResult` items.
   - `POST /api/v1/query`: Full RAG pipeline returning cited, grounded answers with confidence scores and refusal fallback.
3. **Mounted Router in Gateway ([`services/gateway/main.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/main.py))**:
   - Registered `rag_router` into the primary FastAPI application.
4. **Integration Tests ([`tests/unit/test_gateway_rag.py`](file:///d:/Projects/AI-Operations-Platform/tests/unit/test_gateway_rag.py))**:
   - Implemented 5 integration tests covering collection creation, document ingestion, hybrid search, question answering with citations, cascading deletion, and cross-tenant isolation.
   - Test suite now stands at **71 passing tests** with **90% total test coverage**.
   - Mypy passes strictly with 0 errors across 46 source files; Ruff checks 100% clean.

---

### (b) Why we chose this approach
- **Decoupled Gateway Layer via FastAPI Dependencies**: The API Gateway endpoints do not instantiate database connections or vector models directly; they receive them through FastAPI's `Depends` system. This makes unit and integration testing blazingly fast by swapping Qdrant for an in-memory client while keeping production code completely production-ready.
- **Tenant Scope Enforced at Route Boundary**: Every collection name and delete selector derives the Qdrant namespace from `tenant_id` (`col_{tenant_id}_{collection_id}`), preventing accidental cross-tenant data leakage even before queries reach the vector engine.

---

### (c) Alternatives considered and why rejected
- **Direct S3 upload before vectorization for MVP**:
  - *Why rejected*: Storing raw document bytes in S3 will be wired into worker queues in Phase 5. For Phase 1, exposing immediate synchronous ingestion via the Gateway API allows testing the end-to-end RAG pipeline from HTTP request to vector storage immediately.

---

### (d) Trade-offs and risks
- **Synchronous Ingestion on Large Files**: Ingesting 500-page PDFs synchronously over HTTP would cause gateway request timeouts. *Mitigation*: In Phase 5, large document uploads will be queued to RabbitMQ workers with async status polling, while the synchronous endpoint remains for small operational docs and real-time updates.

---

### (e) Non-trivial concepts explained simply

#### 1. FastAPI Dependency Injection (`Depends`)
Instead of global singletons:
```python
# Bad: Hardcoded global client
client = QdrantClient(url="http://localhost:6333")
```
We define dependency functions:
```python
def get_indexer(client: Annotated[QdrantClient, Depends(get_qdrant_client)]) -> QdrantHybridIndexer:
    return QdrantHybridIndexer(client=client)
```
In tests, `app.dependency_overrides[get_qdrant_client] = lambda: QdrantClient(location=":memory:")` instantly swaps the database without changing a single line of production code.

---

### (f) How to verify it works
1. Run gateway RAG integration tests:
   ```bash
   .\.venv\Scripts\pytest.exe tests/unit/test_gateway_rag.py -v
   ```
   *Expected output*: 5 passed in <1.8s.
2. Run full test suite with coverage:
   ```bash
   .\.venv\Scripts\pytest.exe --cov=libs --cov=evals --cov=services -v
   ```
   *Expected output*: 71 passed, 90% total coverage.
3. Run strict linters and type checkers:
   ```bash
   .\.venv\Scripts\ruff.exe check .
   .\.venv\Scripts\mypy.exe libs services evals tests
   ```
   *Expected output*: All checks passed in 46 source files.









---

## [2026-10-07] Phase 1 Completion & Codebase-Wide Educational Documentation Pass

### (a) What was done
1. **Added Comprehensive Educational Comments & Docstrings Across All 21 Source Files**:
   - **`libs/retrieval/`**:
     - [`sparse.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/sparse.py): Documented BM25 term frequency saturation equation, document length normalization factor, $k_1$ and $b$ hyperparameter roles, and the 31-bit MD5 hashing trick for distributed vocabulary-free indexing.
     - [`embeddings.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/embeddings.py): Documented high-dimensional semantic vector spaces, L2 Euclidean normalization (why unit vectors make dot product equivalent to cosine similarity), and deterministic mock vectors.
     - [`indexer.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/indexer.py): Documented dual named vector schema (`"dense"` + `"sparse"`), keyword payload indices for tenant filtering, point upsert batching, and tenant-isolated cascading deletion.
     - [`hybrid.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/hybrid.py): Documented Reciprocal Rank Fusion (RRF) smoothing math ($k=60$), multi-stage retrieval architecture, and small-to-big parent context expansion.
     - [`reranker.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/reranker.py): Documented Cross-Encoder full-attention vs. Bi-Encoder dot product, heuristic token overlap scoring, and Maximal Marginal Relevance (MMR) greedy diversity selection.
     - [`citations.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/citations.py): Documented provenance graph tracking, `[N]` bracket extraction, hallucination detection (`unmapped_citations`), and lexical claim grounding audits.
     - [`generator.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/generator.py): Documented end-to-end RAG pipeline coordination, prompt assembly, and insufficient evidence refusal guards.
     - [`models.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/models.py): Documented domain model separation, small-to-big parent-child chunk modeling (`is_parent`, `parent_id`), and mandatory tenant scope attributes.
     - [`parser.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/parser.py): Documented layout-aware block parsing, heading hierarchy tracking using a section path stack, markdown table buffering, CSV-to-markdown table normalization, and PDF 1-indexed page extraction for citation provenance.
     - [`chunker.py`](file:///d:/Projects/AI-Operations-Platform/libs/retrieval/chunker.py): Documented the chunk size dilemma, recursive splitting with sliding window token overlap, heading-aware splitting, table header repeating across slices, and parent-child hierarchy generation.
   - **`libs/llm/`**:
     - [`provider.py`](file:///d:/Projects/AI-Operations-Platform/libs/llm/provider.py): Documented the Dependency Inversion Principle via `LLMProvider` protocol, LiteLLM automatic fallback routing, and deterministic mock responses.
   - **`evals/`**:
     - [`metrics.py`](file:///d:/Projects/AI-Operations-Platform/evals/metrics.py): Documented Recall@K, MRR, HitRate@K, NDCG@K logarithmic discounting, and lexical faithfulness.
     - [`runner.py`](file:///d:/Projects/AI-Operations-Platform/evals/runner.py): Documented ablation methodology (`dense_only`, `sparse_only`, `hybrid`, `hybrid_rerank`), latency timing, and metric aggregation.
   - **`services/gateway/`**:
     - [`main.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/main.py): Documented Application Factory pattern, ASGI lifespan context managers, middleware ordering (correlation ID before CORS), RFC 7807 error translation, and Kubernetes liveness/readiness probes.
     - [`rag_router.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/rag_router.py): Documented REST API route design, request/response models, and multi-tenant scoping.
     - [`dependencies.py`](file:///d:/Projects/AI-Operations-Platform/services/gateway/dependencies.py): Documented FastAPI dependency injection Directed Acyclic Graphs (DAG), singleton model caching via `@lru_cache`, and in-memory test overrides.
   - **`libs/common/`**:
     - [`config.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/config.py): Documented 12-Factor App configuration via Pydantic `BaseSettings` and zero hardcoded secrets.
     - [`errors.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/errors.py): Documented platform exception hierarchy and RFC 7807 Problem Details serialization.
     - [`health.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/health.py): Documented Kubernetes Liveness vs Readiness semantics, probe timeout isolation, and Prometheus RED metrics.
     - [`logging.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/logging.py): Documented structured JSON logging via `structlog`, task-local `contextvars` propagation, and automatic OpenTelemetry trace/span ID injection.
     - [`telemetry.py`](file:///d:/Projects/AI-Operations-Platform/libs/common/telemetry.py): Documented OpenTelemetry distributed tracing, W3C Trace Context (`traceparent`) injection/extraction across network boundaries, and `CorrelationIdMiddleware`.
2. **Quality Gates Verified**:
   - 71/71 tests passing (100%).
   - 0 syntax or escape sequence warnings.
   - 0 Ruff linting issues.
   - 0 Mypy strict type errors across all 46 source files.

---

### (b) Why we chose this approach
- **Educational Self-Describing Code**: In an enterprise platform with advanced AI concepts (RRF, cross-encoders, vector normalization, W3C trace propagation, RFC 7807 errors), having the explanation and formulas written directly inside each module turns the codebase into a living textbook for Sahil and the team.
- **In-Code Documentation over Stale External Docs**: External wikis and Confluence pages diverge from code within weeks. Placing architectural context and mathematical formulas in docstrings ensures that any future developer modifying the function immediately sees the underlying principles and invariants.

---

### (c) Alternatives considered and why rejected
- **Maintaining documentation solely in external Markdown files**:
  - *Why rejected*: When reading a Python file in an IDE, jumping back and forth to an external markdown document breaks flow. Having rich docstrings allows instant hover-over inspection in VS Code/PyCharm.

---

### (d) Trade-offs and risks
- **File line length increases**: Detailed comments increase source code length by ~40%. *Mitigation*: Python's bytecode compiler strips out comments during execution, and docstrings occupy negligible heap memory, resulting in zero runtime performance penalty.

---

### (e) Non-trivial concepts explained simply

#### 1. Small-to-Big Retrieval (Parent-Child Indexing)
Traditional RAG faces an impossible trade-off:
- If chunks are small (100 words), semantic vector search is accurate, but the LLM answer is incomplete because context was truncated.
- If chunks are large (1,000 words), the LLM has plenty of context, but vector search fails because the 1,000-word chunk is an average of too many ideas.

**The Small-to-Big Solution**:
1. At chunking time: Create a large "Parent Chunk" (2,000 chars), then break it into small "Child Chunks" (500 chars).
2. Index both in Qdrant, but child chunks store `parent_id = "..."`.
3. At search time: Query matches the small child chunk with high vector precision.
4. The retriever looks up `child.parent_id` and swaps the large parent chunk into the LLM prompt!
Result: High vector retrieval accuracy **and** complete context for the LLM.

#### 2. W3C Trace Context Propagation (`traceparent`)
When microservices call each other or publish messages to RabbitMQ, how does Grafana stitch their execution times into a single visual waterfall?
Via the standard W3C `traceparent` header:
`00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`
- `00`: Protocol version.
- `4bf92f3577b34da6a3ce929d0e0e4736`: Global Trace ID (32 hex digits) shared by all services for this request.
- `00f067aa0ba902b7`: Parent Span ID (16 hex digits) representing the caller's specific function span.
- `01`: Trace flags (01 = sampled / recorded).
Our `inject_trace_context()` and `extract_trace_context()` ensure this string is carried across RabbitMQ headers and MCP tool requests.

---

### (f) How to verify it works
1. Run full test suite:
   ```bash
   .\.venv\Scripts\python.exe -m pytest
   ```
   *Expected output*: 71 passed in ~2.2s.
2. Run strict linting and type-checking:
   ```bash
   .\.venv\Scripts\ruff.exe check .
   .\.venv\Scripts\mypy.exe libs services evals tests
   ```
   *Expected output*: All checks passed in 46 source files.
