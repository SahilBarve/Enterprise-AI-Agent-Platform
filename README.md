# Enterprise AI Operations Platform

A production-grade, multi-agent AI operations platform that orchestrates autonomous agents for Document RAG, Web Research, SQL Analytics, Data Processing, and Report Generation with full observability, governance, and automated evaluation.

## Architecture Highlights
- **Multi-Agent Orchestration**: LangGraph supervisor with Planner-Executor-Critic loops and checkpointed state.
- **MCP-First Tool Plane**: Decoupled, independent Model Context Protocol servers for all external tools and databases.
- **HITL Governance**: Risk-tiered, policy-driven approval checkpoints (`interrupt()`) enforced with cryptographically signed single-use tokens.
- **RAG Pipeline**: Hybrid retrieval (dense + sparse BM25 in Qdrant), cross-encoder reranking, two-tier semantic caching, and context compression.
- **Distributed Microservices**: FastAPI, RabbitMQ (`aio-pika`), PostgreSQL, Redis, Docker, Kubernetes (EKS), Terraform.
- **Formal Evaluation**: Ragas + DeepEval CI/CD quality regression gates.

## Quickstart

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### Local Development Setup
1. Clone the repository and initialize the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   pip install -e ".[dev]"
   ```
2. Start the local infrastructure stack:
   ```bash
   docker compose up -d
   ```
3. Run linting and tests:
   ```bash
   ruff check .
   mypy libs services
   pytest
   ```
