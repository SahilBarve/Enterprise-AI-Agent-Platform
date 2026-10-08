"""Integration test verifying Phase 3 Exit Criteria: Multi-step run resumes after crash (FR-OR-5, FR-OR-6).

Scenario:
1. Worker Pod 1 starts a multi-step run requiring human approval.
2. The run hits a Tier 2 action, triggers LangGraph interrupt(), and persists state to SQLite/Postgres.
3. CRASH SIMULATION: Worker Pod 1 is abruptly terminated (engine disposed, memory freed).
4. RECOVERY: A replacement Worker Pod 2 boots up, reads the checkpoint from the database,
   receives the operator's signed approval token, resumes the run from where it stopped,
   and executes to COMPLETED status without re-running prior steps or losing state.
"""

from typing import Any, cast

import pytest

from libs.agents.checkpointer import SQLPlatformCheckpointer
from libs.agents.state import RunStatus
from libs.agents.supervisor import MultiAgentSupervisor
from libs.guardrails.governance import GovernanceGate
from libs.guardrails.tokens import ApprovalTokenManager


@pytest.mark.unit
def test_multi_step_run_resumes_after_crash(tmp_path: pytest.TempPathFactory) -> None:
    """Rigorous verification of Phase 3 Exit Criteria: Multi-step run resumes after crash."""
    db_file = tmp_path / "crash_recovery_test.db"  # type: ignore[operator]
    db_url = f"sqlite:///{db_file}"
    secret_key = "shared-cluster-approval-secret-key"

    run_id = "run-crash-recovery-999"
    tenant_id = "tenant-enterprise-prod"
    goal = "Update settings table in database and generate operational report"

    # =========================================================================
    # Step 1: Worker 1 boots, starts run, halts at Tier 2 interrupt
    # =========================================================================
    worker1_tokens = ApprovalTokenManager(secret_key=secret_key)
    worker1_gate = GovernanceGate(token_manager=worker1_tokens)
    worker1_cp = SQLPlatformCheckpointer(db_url=db_url)

    worker1_supervisor = MultiAgentSupervisor(
        governance_gate=worker1_gate,
        checkpointer=worker1_cp,
    )

    paused_state = worker1_supervisor.start_run(
        goal=goal,
        tenant_id=tenant_id,
        user_id="user-ops",
        run_id=run_id,
    )

    # Confirm run paused at interrupt
    paused_dict = cast(dict[str, Any], paused_state)
    assert "__interrupt__" in paused_dict
    interrupt_payload = paused_dict["__interrupt__"][0].value
    assert interrupt_payload["action"] == "run_sql_write"
    assert interrupt_payload["run_id"] == run_id

    # Verify state snapshot is durably committed to SQL disk
    db_latest = worker1_cp.get_latest_checkpoint(run_id)
    assert db_latest is not None
    assert db_latest.run_id == run_id

    # =========================================================================
    # Step 2: CATASTROPHIC WORKER 1 CRASH SIMULATION
    # Worker 1 container dies; DB connections closed; in-memory supervisor deleted.
    # =========================================================================
    worker1_cp.engine.dispose()
    del worker1_supervisor
    del worker1_cp
    del worker1_gate
    del worker1_tokens

    # =========================================================================
    # Step 3: REPLACEMENT WORKER 2 BOOTS UP AND RESUMES EXECUTION
    # Replacement worker initializes from scratch with no in-memory knowledge.
    # =========================================================================
    worker2_tokens = ApprovalTokenManager(secret_key=secret_key)
    worker2_gate = GovernanceGate(token_manager=worker2_tokens)
    worker2_cp = SQLPlatformCheckpointer(db_url=db_url)

    worker2_supervisor = MultiAgentSupervisor(
        governance_gate=worker2_gate,
        checkpointer=worker2_cp,
    )

    # Worker 2 inspects the crashed run from database checkpoints
    persisted_rec = worker2_supervisor.checkpointer.get_latest_checkpoint(run_id)
    assert persisted_rec is not None
    assert persisted_rec.run_id == run_id
    assert persisted_rec.state["goal"] == goal

    # Operator approves the pending action and issues cryptographically bound token
    approval_token = worker2_tokens.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=interrupt_payload["action"],
        arguments=interrupt_payload["arguments"],
    )

    # Worker 2 resumes the interrupted run!
    resumed_state = worker2_supervisor.resume_run(
        run_id=run_id,
        approval_token=approval_token,
    )

    # =========================================================================
    # Step 4: Verification of Successful Resumption & Completion
    # =========================================================================
    assert resumed_state["status"] == RunStatus.COMPLETED
    assert resumed_state["final_response"] is not None
    assert "successfully executed" in resumed_state["final_response"]

    # Verify all planned steps completed across workers
    plan = resumed_state["plan"]
    assert plan is not None
    assert len(plan.steps) >= 2
    for s in plan.steps:
        assert s.status == "completed"
        assert s.step_id in resumed_state["step_results"]

    # Verify full checkpoint audit trail survived
    all_checkpoints = worker2_cp.list_checkpoints(run_id)
    assert len(all_checkpoints) >= 3
