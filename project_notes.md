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
