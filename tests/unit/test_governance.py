"""Unit tests for risk-tiered policy engine and governance gate (FR-OR-6, FR-SEC-3 to FR-SEC-6)."""

import pytest

from libs.common.errors import GovernanceError
from libs.guardrails.governance import (
    ActionRiskTier,
    GovernanceGate,
    GovernancePolicy,
)
from libs.guardrails.tokens import ApprovalTokenManager


@pytest.fixture
def token_manager() -> ApprovalTokenManager:
    return ApprovalTokenManager(secret_key="test-governance-secret-key")


@pytest.fixture
def governance_gate(token_manager: ApprovalTokenManager) -> GovernanceGate:
    return GovernanceGate(token_manager=token_manager)


@pytest.mark.unit
def test_tier_0_read_only_auto_approved(governance_gate: GovernanceGate) -> None:
    """Validate Tier 0 actions (document retrieval, web search, SQL read) are auto-approved."""
    decision = governance_gate.evaluate(
        run_id="run-1",
        tenant_id="tenant-acme",
        action="retrieve_docs",
        arguments={"collection": "kb", "query": "sla"},
    )
    assert decision.allowed is True
    assert decision.risk_tier == ActionRiskTier.TIER_0_READ_ONLY
    assert decision.requires_approval is False


@pytest.mark.unit
def test_tier_1_low_risk_auto_approved(governance_gate: GovernanceGate) -> None:
    """Validate Tier 1 operations (sandboxed code, drafting) are permitted without approval token."""
    decision = governance_gate.evaluate(
        run_id="run-1",
        tenant_id="tenant-acme",
        action="execute_code_sandbox",
        arguments={"code": "import math; result = math.sqrt(144)"},
    )
    assert decision.allowed is True
    assert decision.risk_tier == ActionRiskTier.TIER_1_LOW_RISK
    assert decision.requires_approval is False


@pytest.mark.unit
def test_tier_2_high_risk_requires_approval_token(
    governance_gate: GovernanceGate, token_manager: ApprovalTokenManager
) -> None:
    """Validate Tier 2 write mutations require a valid HMAC approval token."""
    run_id = "run-sql-44"
    tenant_id = "tenant-acme"
    action = "run_sql_write"
    args = {"sql": "UPDATE subscriptions SET status = 'active' WHERE id = 12;"}

    # 1. Without token -> Rejected (requires_approval=True)
    decision = governance_gate.evaluate(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=args,
        approval_token=None,
    )
    assert decision.allowed is False
    assert decision.risk_tier == ActionRiskTier.TIER_2_HIGH_RISK
    assert decision.requires_approval is True
    assert decision.token_verified is False

    # 2. With valid signed approval token -> Approved (token_verified=True)
    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=args,
    )
    approved_decision = governance_gate.evaluate(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=args,
        approval_token=token,
    )
    assert approved_decision.allowed is True
    assert approved_decision.requires_approval is True
    assert approved_decision.token_verified is True


@pytest.mark.unit
def test_tier_2_tampered_arguments_rejected(
    governance_gate: GovernanceGate, token_manager: ApprovalTokenManager
) -> None:
    """Validate that presenting an approval token with altered arguments is rejected."""
    run_id = "run-tamper-9"
    tenant_id = "tenant-acme"
    action = "run_sql_write"
    approved_args = {"sql": "UPDATE accounts SET balance = 100 WHERE id = 1;"}
    tampered_args = {"sql": "UPDATE accounts SET balance = 999999 WHERE id = 1;"}

    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=approved_args,
    )

    decision = governance_gate.evaluate(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=tampered_args,
        approval_token=token,
    )
    assert decision.allowed is False
    assert "ARGUMENTS_TAMPERED" in decision.reason


@pytest.mark.unit
def test_tier_3_prohibited_actions_blocked(governance_gate: GovernanceGate) -> None:
    """Validate Tier 3 actions (e.g. raw shell, drop db) are unconditionally blocked."""
    decision = governance_gate.evaluate(
        run_id="run-sec-1",
        tenant_id="tenant-acme",
        action="shell_exec",
        arguments={"cmd": "cat /etc/passwd"},
    )
    assert decision.allowed is False
    assert decision.risk_tier == ActionRiskTier.TIER_3_PROHIBITED
    assert "strictly prohibited" in decision.reason


@pytest.mark.unit
def test_enforce_or_raise_behavior(governance_gate: GovernanceGate) -> None:
    """Validate enforce_or_raise raises GovernanceError on policy violations."""
    with pytest.raises(GovernanceError) as exc_info:
        governance_gate.enforce_or_raise(
            run_id="run-1",
            tenant_id="tenant-acme",
            action="drop_database",
            arguments={},
        )
    assert "strictly prohibited" in str(exc_info.value)


@pytest.mark.unit
def test_tenant_specific_policy_overrides(token_manager: ApprovalTokenManager) -> None:
    """Validate custom tenant policies override default risk classifications."""
    # Strict tenant disables all SQL writes and requires approval for sandboxed code
    strict_policy = GovernancePolicy(
        tenant_id="tenant-strict",
        allow_sql_writes=False,
        require_approval_for_all_code=True,
    )
    gate = GovernanceGate(
        token_manager=token_manager,
        tenant_policies={"tenant-strict": strict_policy},
    )

    # 1. Sandboxed code is now Tier 2 instead of Tier 1
    code_decision = gate.evaluate(
        run_id="run-s1",
        tenant_id="tenant-strict",
        action="execute_code_sandbox",
        arguments={"code": "1+1"},
    )
    assert code_decision.risk_tier == ActionRiskTier.TIER_2_HIGH_RISK
    assert code_decision.requires_approval is True

    # 2. SQL write is now Tier 3 (Prohibited) instead of Tier 2
    sql_decision = gate.evaluate(
        run_id="run-s1",
        tenant_id="tenant-strict",
        action="run_sql_write",
        arguments={"sql": "INSERT INTO t VALUES (1)"},
    )
    assert sql_decision.risk_tier == ActionRiskTier.TIER_3_PROHIBITED
    assert sql_decision.allowed is False
