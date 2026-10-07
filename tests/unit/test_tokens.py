"""Unit tests for cryptographic argument-bound approval tokens (FR-OR-6, FR-SEC-5, P0 Pillar)."""

import pytest

from libs.common.errors import GovernanceError
from libs.guardrails.tokens import (
    ApprovalTokenManager,
    canonical_json,
    hash_arguments,
)


@pytest.fixture
def token_manager() -> ApprovalTokenManager:
    return ApprovalTokenManager(secret_key="test-secure-hmac-key-for-testing-only")


@pytest.mark.unit
def test_canonical_json_and_argument_hashing() -> None:
    """Validate deterministic hashing irrespective of key ordering in dictionary."""
    dict1 = {"action": "drop_table", "table": "orders", "force": True}
    dict2 = {"force": True, "table": "orders", "action": "drop_table"}

    assert canonical_json(dict1) == canonical_json(dict2)
    assert hash_arguments(dict1) == hash_arguments(dict2)

    # Different arguments produce different hashes
    dict3 = {"force": False, "table": "orders", "action": "drop_table"}
    assert hash_arguments(dict1) != hash_arguments(dict3)


@pytest.mark.unit
def test_issue_and_verify_valid_token(token_manager: ApprovalTokenManager) -> None:
    """Validate issuing and verifying an authentic approval token."""
    run_id = "run-prod-001"
    tenant_id = "tenant-corp"
    action = "run_sql_write"
    args = {"sql": "UPDATE users SET active = true WHERE id = 10;"}

    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=args,
        ttl_seconds=300,
    )

    assert isinstance(token, str)
    assert "." in token

    result = token_manager.verify_and_consume(
        token_str=token,
        expected_run_id=run_id,
        expected_tenant_id=tenant_id,
        expected_action=action,
        actual_arguments=args,
    )

    assert result.is_valid is True
    assert result.payload is not None
    assert result.payload.run_id == run_id
    assert result.payload.action == action


@pytest.mark.unit
def test_anti_tampering_argument_modification(token_manager: ApprovalTokenManager) -> None:
    """Validate that altering even one character in approved arguments fails verification."""
    run_id = "run-prod-002"
    tenant_id = "tenant-corp"
    action = "run_sql_write"
    approved_args = {"sql": "UPDATE users SET active = true WHERE id = 10;"}

    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=approved_args,
    )

    # Attacker or prompt injection alters the SQL to a destructive query
    tampered_args = {"sql": "DROP TABLE users;"}

    result = token_manager.verify_and_consume(
        token_str=token,
        expected_run_id=run_id,
        expected_tenant_id=tenant_id,
        expected_action=action,
        actual_arguments=tampered_args,
    )

    assert result.is_valid is False
    assert result.error_code == "ARGUMENTS_TAMPERED"


@pytest.mark.unit
def test_single_use_replay_attack_prevention(token_manager: ApprovalTokenManager) -> None:
    """Validate that an approval token cannot be replayed twice."""
    run_id = "run-prod-003"
    tenant_id = "tenant-corp"
    action = "send_webhook"
    args = {"url": "https://api.partner.com/notify", "event": "order_completed"}

    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=action,
        arguments=args,
    )

    # First execution succeeds
    res1 = token_manager.verify_and_consume(
        token_str=token,
        expected_run_id=run_id,
        expected_tenant_id=tenant_id,
        expected_action=action,
        actual_arguments=args,
    )
    assert res1.is_valid is True

    # Second execution attempt is rejected as already consumed
    res2 = token_manager.verify_and_consume(
        token_str=token,
        expected_run_id=run_id,
        expected_tenant_id=tenant_id,
        expected_action=action,
        actual_arguments=args,
    )
    assert res2.is_valid is False
    assert res2.error_code == "TOKEN_ALREADY_CONSUMED"


@pytest.mark.unit
def test_context_mismatch_detection(token_manager: ApprovalTokenManager) -> None:
    """Validate detection of run ID, tenant ID, or action mismatches."""
    args = {"query": "DELETE FROM sessions"}
    token = token_manager.issue_token(
        run_id="run-A",
        tenant_id="tenant-1",
        action="run_sql_write",
        arguments=args,
    )

    # Run ID mismatch
    res_run = token_manager.verify_and_consume(
        token, expected_run_id="run-B", expected_tenant_id="tenant-1", expected_action="run_sql_write", actual_arguments=args
    )
    assert res_run.error_code == "RUN_ID_MISMATCH"

    # Tenant mismatch
    res_tenant = token_manager.verify_and_consume(
        token, expected_run_id="run-A", expected_tenant_id="tenant-2", expected_action="run_sql_write", actual_arguments=args
    )
    assert res_tenant.error_code == "TENANT_MISMATCH"

    # Action mismatch
    res_act = token_manager.verify_and_consume(
        token, expected_run_id="run-A", expected_tenant_id="tenant-1", expected_action="send_webhook", actual_arguments=args
    )
    assert res_act.error_code == "ACTION_MISMATCH"


@pytest.mark.unit
def test_token_expiration(token_manager: ApprovalTokenManager) -> None:
    """Validate that expired tokens are rejected."""
    args = {"file": "export.csv"}
    # Issue token with negative TTL (already expired)
    token = token_manager.issue_token(
        run_id="run-exp",
        tenant_id="tenant-corp",
        action="publish_report",
        arguments=args,
        ttl_seconds=-10,
    )

    res = token_manager.verify_and_consume(
        token, expected_run_id="run-exp", expected_tenant_id="tenant-corp", expected_action="publish_report", actual_arguments=args
    )
    assert res.is_valid is False
    assert res.error_code == "TOKEN_EXPIRED"


@pytest.mark.unit
def test_signature_tampering_and_malformed_tokens(token_manager: ApprovalTokenManager) -> None:
    """Validate rejection of forged signatures and malformed token strings."""
    args = {"key": "val"}
    token = token_manager.issue_token("run-x", "tenant-x", "action-x", args)

    payload_b64, sig = token.split(".")
    # Tamper with signature
    tampered_sig = sig[:-4] + "AAAA"
    bad_token = f"{payload_b64}.{tampered_sig}"

    res = token_manager.verify_and_consume(
        bad_token, "run-x", "tenant-x", "action-x", args
    )
    assert res.is_valid is False
    assert res.error_code == "INVALID_SIGNATURE"

    # Malformed format (no dot)
    res_malformed = token_manager.verify_and_consume(
        "no-dot-token", "run-x", "tenant-x", "action-x", args
    )
    assert res_malformed.error_code == "INVALID_TOKEN_FORMAT"


@pytest.mark.unit
def test_enforce_token_or_raise_helper(token_manager: ApprovalTokenManager) -> None:
    """Validate enforce_token_or_raise helper raises GovernanceError on invalid tokens."""
    args = {"val": 1}
    with pytest.raises(GovernanceError) as exc_info:
        token_manager.enforce_token_or_raise(
            token_str=None,
            expected_run_id="run-1",
            expected_tenant_id="tenant-1",
            expected_action="run_sql_write",
            actual_arguments=args,
        )
    assert "requires a signed approval token" in str(exc_info.value)
