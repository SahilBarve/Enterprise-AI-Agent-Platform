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


