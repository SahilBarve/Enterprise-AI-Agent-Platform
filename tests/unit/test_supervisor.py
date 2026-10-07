from typing import Any, cast

import pytest

from libs.agents.checkpointer import MemoryPlatformCheckpointer
from libs.agents.state import BudgetLimits, RunStatus
from libs.agents.supervisor import MultiAgentSupervisor
from libs.common.errors import GovernanceError
from libs.guardrails.governance import GovernanceGate
from libs.guardrails.tokens import ApprovalTokenManager


@pytest.fixture
def supervisor() -> MultiAgentSupervisor:
    token_manager = ApprovalTokenManager(secret_key="test-supervisor-secret-key-32chars")
    governance_gate = GovernanceGate(token_manager=token_manager)
    checkpointer = MemoryPlatformCheckpointer()
    return MultiAgentSupervisor(
        governance_gate=governance_gate,
        checkpointer=checkpointer,
    )


@pytest.mark.unit
def test_supervisor_autonomous_read_run(supervisor: MultiAgentSupervisor) -> None:
    """Validate autonomous multi-step execution for safe read-only goals (Tier 0)."""
    goal = "Retrieve internal architecture documents and query analytics database"
    run_id = "run-auto-101"
    tenant_id = "tenant-corp"

    state = supervisor.start_run(
        goal=goal,
        tenant_id=tenant_id,
        user_id="user-analyst",
        run_id=run_id,
    )

    # Validate plan was executed and completed
    assert state["status"] == RunStatus.COMPLETED
    assert state["plan"] is not None
    assert len(state["plan"].steps) >= 2
    assert state["final_response"] is not None
    assert "successfully executed" in state["final_response"]

    # Validate timeline events
    event_types = [e.event_type for e in state["events"]]
    assert "plan_created" in event_types
    assert "step_complete" in event_types
    assert "run_completed" in event_types

    # Validate checkpoints were persisted
    history = supervisor.checkpointer.list_checkpoints(run_id)
    assert len(history) >= 2


@pytest.mark.unit
def test_supervisor_hitl_interrupt_and_resume(supervisor: MultiAgentSupervisor) -> None:
    """Validate that Tier 2 write mutations interrupt the graph and resume with signed token (FR-OR-6, P0 Pillar)."""
    # A goal requesting an update triggers a Tier 2 action in the planner
    goal = "Update settings table in database and delete old logs"
    run_id = "run-hitl-202"
    tenant_id = "tenant-corp"

    # 1. Start run: Execution reaches Tier 2 action and triggers interrupt()
    paused_state = supervisor.start_run(
        goal=goal,
        tenant_id=tenant_id,
        user_id="user-admin",
        run_id=run_id,
    )

    # Validate that execution paused at interrupt
    paused_dict = cast(dict[str, Any], paused_state)
    assert "__interrupt__" in paused_dict
    interrupt_info = paused_dict["__interrupt__"][0]
    appr_payload = interrupt_info.value
    assert appr_payload["action"] == "run_sql_write"
    assert appr_payload["risk_tier"] == 2
    assert "sql" in appr_payload["arguments"]

    # 2. Operator reviews and issues signed HMAC approval token
    token = supervisor.governance_gate.token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=appr_payload["action"],
        arguments=appr_payload["arguments"],
    )

    # 3. Resume the run using the signed approval token
    resumed_state = supervisor.resume_run(run_id=run_id, approval_token=token)

    # 4. Verify the run resumed from the checkpoint and completed
    assert resumed_state["status"] == RunStatus.COMPLETED
    assert resumed_state["final_response"] is not None
    assert "successfully executed" in resumed_state["final_response"]


@pytest.mark.unit
def test_supervisor_hitl_tampered_token_rejected(supervisor: MultiAgentSupervisor) -> None:
    """Validate that presenting an invalid or tampered approval token fails verification upon resume."""
    goal = "Update settings table in database"
    run_id = "run-tamper-303"
    tenant_id = "tenant-corp"

    paused_state = supervisor.start_run(
        goal=goal,
        tenant_id=tenant_id,
        user_id="user-admin",
        run_id=run_id,
    )
    paused_dict = cast(dict[str, Any], paused_state)
    assert "__interrupt__" in paused_dict

    # Issue token for completely different SQL arguments
    tampered_token = supervisor.governance_gate.token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action="run_sql_write",
        arguments={"sql": "DROP TABLE critical_data;"},
    )

    # Resuming with token that does NOT match the pending arguments must raise GovernanceError
    with pytest.raises(GovernanceError) as exc_info:
        supervisor.resume_run(run_id=run_id, approval_token=tampered_token)

    assert "approval token verification failed" in str(exc_info.value).lower()


@pytest.mark.unit
def test_supervisor_budget_guard_exceeded(supervisor: MultiAgentSupervisor) -> None:
    """Validate that exceeding configured budget limits halts execution with FAILED status (FR-OR-7)."""
    goal = "Retrieve internal documents and query analytics database and generate report"
    run_id = "run-budget-404"
    tenant_id = "tenant-corp"

    # Set very tight budget: max 1 step
    tight_limits = BudgetLimits(max_steps=1)

    state = supervisor.start_run(
        goal=goal,
        tenant_id=tenant_id,
        user_id="user-test",
        run_id=run_id,
        budget_limits=tight_limits,
    )

    # The run should halt after step 1 when the router checks budget for step 2
    assert state["status"] == RunStatus.FAILED
    assert "Exceeded maximum steps" in str(state.get("error"))
