"""Integration tests for Gateway RAG endpoints (FR-GW-1, FR-RAG-1, FR-RAG-10, FR-RAG-11, FR-RAG-26)."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from libs.common.config import PlatformSettings
from services.gateway.dependencies import get_qdrant_client
from services.gateway.main import create_gateway_app


@pytest.fixture
def in_memory_qdrant() -> QdrantClient:
    return QdrantClient(location=":memory:")


@pytest.fixture
def client(in_memory_qdrant: QdrantClient) -> TestClient:
    settings = PlatformSettings(ENVIRONMENT="testing", LOG_LEVEL="DEBUG")
    app = create_gateway_app(settings)
    # Override Qdrant client to use in-memory instance
    app.dependency_overrides[get_qdrant_client] = lambda: in_memory_qdrant
    return TestClient(app)


@pytest.mark.unit
def test_create_collection_endpoint(client: TestClient) -> None:
    """Validate POST /api/v1/collections creates a tenant vector collection."""
    res = client.post(
        "/api/v1/collections",
        json={"tenant_id": "tenant-test", "collection_id": "test-docs"},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["collection_name"] == "col_tenant_test_test_docs"
    assert data["status"] == "created"

    # Second call returns already_exists
    res2 = client.post(
        "/api/v1/collections",
        json={"tenant_id": "tenant-test", "collection_id": "test-docs"},
    )
    assert res2.status_code == 201
    assert res2.json()["status"] == "already_exists"


@pytest.mark.unit
def test_ingest_and_search_endpoints(client: TestClient) -> None:
    """Validate document ingestion, chunking, indexing, and hybrid search via API."""
    tenant_id = "tenant-ops"
    collection_id = "ops-kb"

    # 1. Ingest document
    doc_payload = {
        "tenant_id": tenant_id,
        "title": "Database Policy",
        "content": (
            "# Database Scaling\n\n"
            "PostgreSQL read replicas are scaled with streaming replication.\n\n"
            "Patroni monitors cluster health and manages automated leader failover."
        ),
        "content_type": "text/markdown",
        "chunk_strategy": "heading_aware",
        "chunk_size": 300,
        "metadata": {"department": "infrastructure"},
    }

    ingest_res = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        json=doc_payload,
    )
    assert ingest_res.status_code == 201
    ingest_data = ingest_res.json()
    assert "document_id" in ingest_data
    assert ingest_data["chunks_indexed"] > 0
    assert len(ingest_data["content_hash"]) == 64

    # 2. Search document
    search_payload = {
        "query": "Patroni automated failover",
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "top_k": 3,
        "rerank": True,
    }
    search_res = client.post("/api/v1/search", json=search_payload)
    assert search_res.status_code == 200
    hits = search_res.json()
    assert len(hits) > 0
    top_hit = hits[0]
    assert top_hit["tenant_id"] == tenant_id
    assert "Patroni" in top_hit["content"]
    assert top_hit["score"] > 0.0


@pytest.mark.unit
def test_query_grounded_answer_endpoint(client: TestClient) -> None:
    """Validate POST /api/v1/query returns cited and grounded response (FR-RAG-26)."""
    tenant_id = "tenant-ops"
    collection_id = "ops-kb"

    # Ingest document
    client.post(
        f"/api/v1/collections/{collection_id}/documents",
        json={
            "tenant_id": tenant_id,
            "title": "Kubernetes Autoscaling",
            "content": "KEDA scales worker pods based on RabbitMQ queue depth with 100 messages target per pod.",
            "chunk_strategy": "recursive",
        },
    )

    query_res = client.post(
        "/api/v1/query",
        json={
            "query": "How does KEDA scale worker pods?",
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "top_k": 2,
        },
    )
    assert query_res.status_code == 200
    answer = query_res.json()
    assert "answer" in answer
    assert "citations" in answer
    assert answer["confidence_score"] > 0.0
    assert answer["is_refusal"] is False


@pytest.mark.unit
def test_delete_document_endpoint(client: TestClient) -> None:
    """Validate DELETE /api/v1/collections/{id}/documents/{id} removes points (FR-RAG-10)."""
    tenant_id = "tenant-temp"
    collection_id = "temp-kb"

    # Ingest
    ingest_res = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        json={
            "tenant_id": tenant_id,
            "title": "Ephemeral Note",
            "content": "This note will be deleted immediately.",
        },
    )
    doc_id = ingest_res.json()["document_id"]

    # Delete
    del_res = client.delete(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}?tenant_id={tenant_id}"
    )
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Search should yield 0 hits
    search_res = client.post(
        "/api/v1/search",
        json={
            "query": "Ephemeral Note",
            "tenant_id": tenant_id,
            "collection_id": collection_id,
        },
    )
    assert len(search_res.json()) == 0


@pytest.mark.unit
def test_cross_tenant_isolation_via_api(client: TestClient) -> None:
    """Validate tenant A cannot search tenant B documents via API (FR-GW-3)."""
    # Tenant Alpha
    client.post(
        "/api/v1/collections/common/documents",
        json={
            "tenant_id": "tenant-alpha",
            "title": "Alpha Secret",
            "content": "Alpha proprietary algorithm secret key is 998877.",
        },
    )

    # Tenant Beta search against common collection
    res_beta = client.post(
        "/api/v1/search",
        json={
            "query": "proprietary algorithm secret key",
            "tenant_id": "tenant-beta",
            "collection_id": "common",
        },
    )
    assert res_beta.status_code == 200
    # Beta's isolated collection does not contain Alpha's document
    assert len(res_beta.json()) == 0
