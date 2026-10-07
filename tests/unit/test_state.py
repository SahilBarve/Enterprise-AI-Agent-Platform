"""Unit tests for LangGraph state schema, budget guards, and reducers (FR-OR-2, FR-OR-7, FR-OR-9)."""

from datetime import UTC, datetime, timedelta

import pytest

from libs.agents.state import (
    AgentType,
    ApprovalRequest,
    BudgetLimits,
    BudgetUsage,
    Plan,
    PlanStep,
    RunStatus,
    StepActionType,
    StepEvent,
    append_artifacts,
    append_events,
    merge_step_results,
)


@pytest.mark.unit
def test_run_status_and_enums() -> None:
    """Validate all required lifecycle states and agent types are defined."""
    assert RunStatus.QUEUED.value == "queued"
    assert RunStatus.PLANNING.value == "planning"
    assert RunStatus.RUNNING.value == "running"
    assert RunStatus.AWAITING_APPROVAL.value == "awaiting_approval"
    assert RunStatus.COMPLETED.value == "completed"
    assert RunStatus.FAILED.value == "failed"
    assert RunStatus.CANCELLED.value == "cancelled"

    assert AgentType.SUPERVISOR.value == "supervisor"
    assert AgentType.DOCUMENT_RAG.value == "document_rag"
    assert AgentType.WEB_RESEARCH.value == "web_research"
    assert AgentType.SQL_ANALYTICS.value == "sql_analytics"
    assert AgentType.DATA_PROCESSING.value == "data_processing"
    assert AgentType.REPORT_GENERATION.value == "report_generation"

    assert StepActionType.RUN_SQL_WRITE.value == "run_sql_write"
    assert StepActionType.RETRIEVE_DOCS.value == "retrieve_docs"


@pytest.mark.unit
def test_plan_and_plan_step_modeling() -> None:
    """Validate plan step creation, dependencies, and index tracking."""
    step1 = PlanStep(
        title="Retrieve Architecture Docs",
        description="Search for microservices communication patterns",
        agent=AgentType.DOCUMENT_RAG,
        action_type=StepActionType.RETRIEVE_DOCS,
        action_arguments={"collection": "architecture", "query": "gRPC protocols"},
    )
    step2 = PlanStep(
        title="Execute Write Migration",
        description="Apply database schema updates",
        agent=AgentType.SQL_ANALYTICS,
        action_type=StepActionType.RUN_SQL_WRITE,
        action_arguments={"sql": "ALTER TABLE users ADD COLUMN tier VARCHAR(32);"},
        dependencies=[step1.step_id],
    )

    plan = Plan(
        goal="Update schema after verifying architecture documentation",
        steps=[step1, step2],
        current_step_index=0,
    )

    assert len(plan.steps) == 2
    assert plan.steps[1].dependencies == [step1.step_id]
    assert plan.is_replan is False
    assert plan.steps[0].status == "pending"


@pytest.mark.unit
def test_budget_usage_and_limit_enforcement() -> None:
    """Validate multi-agent budget guards against step, token, cost, and time limits."""
    limits = BudgetLimits(
        max_steps=10,
        max_tokens=1000,
        max_cost_usd=1.50,
        max_wall_clock_seconds=60.0,
    )

    # 1. Nominal usage within limits
    usage = BudgetUsage(
        steps_taken=5,
        tokens_used=500,
        cost_usd=0.75,
        start_time=datetime.now(UTC),
    )
    exceeded, _ = usage.is_exceeded(limits)
    assert exceeded is False

    # 2. Exceeded steps
    usage.steps_taken = 10
    exceeded, reason = usage.is_exceeded(limits)
    assert exceeded is True
    assert "Exceeded maximum steps" in reason

    # 3. Exceeded tokens
    usage.steps_taken = 5
    usage.tokens_used = 1200
    exceeded, reason = usage.is_exceeded(limits)
    assert exceeded is True
    assert "Exceeded maximum tokens" in reason

    # 4. Exceeded cost
    usage.tokens_used = 500
    usage.cost_usd = 2.00
    exceeded, reason = usage.is_exceeded(limits)
    assert exceeded is True
    assert "Exceeded maximum cost" in reason

    # 5. Exceeded wall-clock time
    usage.cost_usd = 0.50
    usage.start_time = datetime.now(UTC) - timedelta(seconds=70)
    exceeded, reason = usage.is_exceeded(limits)
    assert exceeded is True
    assert "Exceeded time limit" in reason


@pytest.mark.unit
def test_state_reducers() -> None:
    """Validate LangGraph reducers append events and artifacts without overwriting."""
    # Test append_events reducer
    e1 = StepEvent(run_id="run-1", event_type="plan_created", agent="supervisor")
    e2 = StepEvent(run_id="run-1", event_type="step_start", agent="document_rag")
    e3 = StepEvent(run_id="run-1", event_type="step_complete", agent="document_rag")

    events = append_events([e1], [e2, e3])
    assert len(events) == 3
    assert events[0].event_type == "plan_created"
    assert events[2].event_type == "step_complete"

    # Reducer handles None left or right gracefully
    assert len(append_events(None, [e1])) == 1
    assert len(append_events([e1], None)) == 1

    # Test append_artifacts reducer
    a1 = {"name": "report.pdf", "size_kb": 240}
    a2 = {"name": "metrics.png", "size_kb": 85}
    artifacts = append_artifacts([a1], [a2])
    assert len(artifacts) == 2
    assert artifacts[0]["name"] == "report.pdf"
    assert artifacts[1]["name"] == "metrics.png"

    # Test merge_step_results reducer
    r1 = {"step-1": {"status": "success", "count": 42}}
    r2 = {"step-2": {"status": "success", "revenue": 10500}}
    merged = merge_step_results(r1, r2)
    assert "step-1" in merged
    assert "step-2" in merged
    assert merged["step-2"]["revenue"] == 10500


@pytest.mark.unit
def test_approval_request_modeling() -> None:
    """Validate ApprovalRequest structure for governance interrupts."""
    req = ApprovalRequest(
        run_id="run-test-99",
        tenant_id="tenant-acme",
        step_id="step-write-1",
        action="run_sql_write",
        arguments={"sql": "DROP TABLE temp_logs;"},
        reason="Destructive DDL requires Tier 2 approval",
        risk_tier=2,
    )
    assert req.approved is None
    assert req.risk_tier == 2
    assert req.approval_token is None

    # Simulate sign-off
    req.approved = True
    req.approved_by = "admin@acme.corp"
    req.approval_token = "token-hmac-signed-xyz"
    assert req.approved is True
    assert req.approved_by == "admin@acme.corp"
