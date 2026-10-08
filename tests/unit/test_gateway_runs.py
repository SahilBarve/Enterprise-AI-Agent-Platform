"""Unit tests for Gateway orchestrator runs and approval endpoints (FR-OR-1, FR-OR-6, FR-GW-6)."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from libs.agents.state import RunStatus
from libs.common.config import PlatformSettings
from services.gateway.dependencies import get_qdrant_client
from services.gateway.main import create_gateway_app


@pytest.fixture
def client() -> TestClient:
    settings = PlatformSettings(ENVIRONMENT="testing", LOG_LEVEL="DEBUG")
    app = create_gateway_app(settings)
    app.dependency_overrides[get_qdrant_client] = lambda: QdrantClient(location=":memory:")
    return TestClient(app)


@pytest.mark.unit
def test_create_and_get_autonomous_run(client: TestClient) -> None:
    """Validate creating an autonomous run and inspecting its state."""
    payload = {
        "goal": "Retrieve internal architecture documents and query analytics database",
        "tenant_id": "tenant-acme",
        "user_id": "user-sahil",
        "max_steps": 10,
    }
    resp = client.post("/api/v1/runs", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    run_id = data["run_id"]
    assert data["status"] == RunStatus.COMPLETED.value
    assert data["goal"] == payload["goal"]
    assert data["plan"] is not None
    assert len(data["plan"]["steps"]) >= 2
    assert "successfully executed" in data["final_response"]

    # Fetch run details by ID
    get_resp = client.get(f"/api/v1/runs/{run_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["run_id"] == run_id
    assert get_data["status"] == RunStatus.COMPLETED.value


@pytest.mark.unit
def test_run_timeline_events_and_sse_streaming(client: TestClient) -> None:
    """Validate fetching events array and SSE streaming format (FR-GW-6)."""
    create_resp = client.post(
        "/api/v1/runs",
        json={"goal": "Retrieve documentation", "tenant_id": "tenant-corp"},
    )
    run_id = create_resp.json()["run_id"]

    # 1. Standard JSON list
    events_resp = client.get(f"/api/v1/runs/{run_id}/events")
    assert events_resp.status_code == 200
    events = events_resp.json()
    assert isinstance(events, list)
    assert len(events) >= 2
    event_types = [e["event_type"] for e in events]
    assert "plan_created" in event_types

    # 2. SSE streaming format
    sse_resp = client.get(f"/api/v1/runs/{run_id}/events?stream=true")
    assert sse_resp.status_code == 200
    assert "text/event-stream" in sse_resp.headers["content-type"]
    assert "data: " in sse_resp.text
    assert "[DONE]" in sse_resp.text


@pytest.mark.unit
def test_hitl_approval_workflow_via_api(client: TestClient) -> None:
    """Validate end-to-end human-in-the-loop pause and resume over HTTP (FR-OR-6, P0 Pillar)."""
    # Write mutation goal triggers Tier 2 interrupt
    payload = {
        "goal": "Update settings table in database and delete old logs",
        "tenant_id": "tenant-acme",
    }
    resp = client.post("/api/v1/runs", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    run_id = data["run_id"]
    assert data["pending_approval"] is not None
    pending = data["pending_approval"]
    assert pending["action"] == "run_sql_write"
    assert pending["risk_tier"] == 2
    assert "sql" in pending["arguments"]

    # Submit operator sign-off via POST /runs/{id}/approve
    appr_payload = {
        "action": pending["action"],
        "arguments": pending["arguments"],
        "approved": True,
        "approver_id": "sec-admin@acme.corp",
        "notes": "Verified maintenance mode change window",
    }
    appr_resp = client.post(f"/api/v1/runs/{run_id}/approve", json=appr_payload)
    assert appr_resp.status_code == 200
    resumed_data = appr_resp.json()
    assert resumed_data["status"] == RunStatus.COMPLETED.value
    assert "successfully executed" in resumed_data["final_response"]


@pytest.mark.unit
def test_hitl_rejection_workflow_via_api(client: TestClient) -> None:
    """Validate operator rejecting a high-risk mutation cancels the run."""
    resp = client.post(
        "/api/v1/runs",
        json={"goal": "Update settings table in database", "tenant_id": "tenant-acme"},
    )
    run_id = resp.json()["run_id"]

    reject_payload = {
        "action": "run_sql_write",
        "arguments": {"sql": "UPDATE settings SET maintenance = true WHERE id = 1;"},
        "approved": False,
        "approver_id": "sec-admin@acme.corp",
    }
    reject_resp = client.post(f"/api/v1/runs/{run_id}/approve", json=reject_payload)
    assert reject_resp.status_code == 200
    assert reject_resp.json()["status"] == RunStatus.CANCELLED.value


@pytest.mark.unit
def test_checkpoints_and_forking_via_api(client: TestClient) -> None:
    """Validate listing checkpoints and time-travel forking via API (FR-OR-5, FR-OR-10)."""
    resp = client.post(
        "/api/v1/runs",
        json={"goal": "Retrieve documentation", "tenant_id": "tenant-acme"},
    )
    run_id = resp.json()["run_id"]

    # List checkpoints
    chk_resp = client.get(f"/api/v1/runs/{run_id}/checkpoints")
    assert chk_resp.status_code == 200
    checkpoints = chk_resp.json()
    assert len(checkpoints) >= 1
    chk_id = checkpoints[0]["checkpoint_id"]

    # Fork from checkpoint into new run
    fork_payload = {
        "source_checkpoint_id": chk_id,
        "new_run_id": f"fork-{run_id}",
        "state_overrides": {"note": "time-travel branch"},
    }
    fork_resp = client.post(f"/api/v1/runs/{run_id}/fork", json=fork_payload)
    assert fork_resp.status_code == 201
    assert fork_resp.json()["run_id"] == f"fork-{run_id}"
