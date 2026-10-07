"""HMAC-SHA256 cryptographically bound, single-use approval tokens (FR-OR-6, FR-SEC-5, P0 Pillar).

Architectural Concepts Explained:
--------------------------------
1. Why Simple Approval Flags Fail in Production:
   - In naive implementations, human approvals are represented by a database column `is_approved = True`.
   - Vulnerability 1 (Argument Tampering / TOCTOU):
     An admin inspects `UPDATE orders SET status = 'shipped' WHERE id = 42` and clicks Approve.
     A race condition or malicious prompt injection alters the pending query in the database to
     `DROP TABLE orders;` before execution. The naive worker sees `is_approved == True` and executes it!
   - Vulnerability 2 (Replay Attacks):
     An approved action is executed multiple times because the approval state is not atomically invalidated.

2. Cryptographic Argument Binding (The Solution):
   - When approval is granted, the platform computes a SHA-256 digest over the canonical JSON arguments:
     PayloadHash = SHA256(canonical_json(arguments))
   - A cryptographic HMAC signature is computed across (token_id, run_id, tenant_id, action, PayloadHash, expires_at).
   - Before executing the dangerous action, the worker recalculates PayloadHash from the *actual* parameters
     passed to the tool. If even a single character was modified, the hash check fails!

3. Single-Use Replay Protection:
   - Every token carries a unique UUID (`token_id`).
   - Upon successful verification, `token_id` is atomically recorded in a consumed-token store.
   - Any second attempt to execute using the same token is rejected with `TOKEN_ALREADY_CONSUMED`.
"""

import hashlib
import hmac
import json
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from libs.common.config import get_settings
from libs.common.errors import GovernanceError
from libs.common.logging import get_logger

logger = get_logger("guardrails.tokens")


def canonical_json(data: dict[str, Any]) -> str:
    """Serialize dictionary to deterministic canonical JSON (sorted keys, compact separators).

    Ensures that identical argument dictionaries always yield the exact same SHA-256 hash.
    """

    def default_encoder(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return str(o)

    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=default_encoder)


def hash_arguments(arguments: dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of canonical arguments dictionary."""
    canonical_str = canonical_json(arguments)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class ApprovalTokenPayload(BaseModel):
    """Payload embedded inside a signed approval token."""

    token_id: str = Field(
        default_factory=lambda: f"tok-{uuid4().hex[:16]}",
        description="Unique single-use token UUID",
    )
    run_id: str = Field(description="Orchestrator run ID")
    tenant_id: str = Field(description="Tenant scope")
    action: str = Field(description="Target action (e.g. 'run_sql_write')")
    payload_hash: str = Field(description="SHA-256 hash of canonical arguments")
    expires_at: int = Field(description="Unix timestamp when token expires")
    created_at: int = Field(description="Unix timestamp when token was issued")


class VerificationResult(BaseModel):
    """Result of token verification check."""

    is_valid: bool
    payload: ApprovalTokenPayload | None = None
    error_code: str | None = None
    error_message: str | None = None


class ApprovalTokenManager:
    """Manages creation, signature, verification, and single-use consumption of approval tokens."""

    def __init__(self, secret_key: str | None = None) -> None:
        """Initialize token manager.

        Args:
            secret_key: Secret key for HMAC-SHA256. If None, uses settings.APPROVAL_TOKEN_SECRET.
        """
        self.secret_key = secret_key or get_settings().APPROVAL_TOKEN_SECRET
        self._consumed_token_ids: set[str] = set()

    def _sign(self, signing_input: str) -> str:
        """Generate HMAC-SHA256 signature for the input string."""
        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return urlsafe_b64encode(sig).decode("utf-8").rstrip("=")

    def issue_token(
        self,
        run_id: str,
        tenant_id: str,
        action: str,
        arguments: dict[str, Any],
        ttl_seconds: int = 1800,
    ) -> str:
        """Issue a signed, argument-bound approval token.

        Args:
            run_id: Active orchestrator run ID.
            tenant_id: Tenant scope.
            action: Action being authorized (e.g. 'run_sql_write', 'send_webhook').
            arguments: Exact dictionary of arguments to bind to the token.
            ttl_seconds: Token validity duration (default: 30 minutes).

        Returns:
            Opaque URL-safe base64-encoded token string.
        """
        now = int(datetime.now(UTC).timestamp())
        payload = ApprovalTokenPayload(
            run_id=run_id,
            tenant_id=tenant_id,
            action=action,
            payload_hash=hash_arguments(arguments),
            expires_at=now + ttl_seconds,
            created_at=now,
        )

        payload_json = payload.model_dump_json()
        payload_b64 = urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
        signature = self._sign(payload_b64)

        token_str = f"{payload_b64}.{signature}"
        logger.info(
            "Issued cryptographic approval token",
            token_id=payload.token_id,
            run_id=run_id,
            tenant_id=tenant_id,
            action=action,
            expires_at=payload.expires_at,
        )
        return token_str

    def verify_and_consume(
        self,
        token_str: str,
        expected_run_id: str,
        expected_tenant_id: str,
        expected_action: str,
        actual_arguments: dict[str, Any],
    ) -> VerificationResult:
        """Verify token cryptographic signature, argument binding, expiration, and consume it.

        Checks:
        1. Token structure (payload.signature).
        2. HMAC-SHA256 signature validity.
        3. Token expiration timestamp against current UTC time.
        4. Matching run_id, tenant_id, and action.
        5. Argument hash match against actual arguments passed at execution time (tamper check).
        6. Single-use replay protection: token_id has not been consumed previously.

        Returns:
            VerificationResult indicating success or specific validation failure.
        """
        parts = token_str.strip().split(".")
        if len(parts) != 2:
            return VerificationResult(
                is_valid=False,
                error_code="INVALID_TOKEN_FORMAT",
                error_message="Token must have payload.signature structure",
            )

        payload_b64, signature = parts[0], parts[1]

        # 1. Verify HMAC signature
        expected_sig = self._sign(payload_b64)
        if not hmac.compare_digest(signature, expected_sig):
            logger.warning("Approval token signature mismatch / tampering detected")
            return VerificationResult(
                is_valid=False,
                error_code="INVALID_SIGNATURE",
                error_message="Cryptographic signature verification failed",
            )

        # 2. Decode and parse payload
        try:
            # Re-pad base64
            padded_b64 = payload_b64 + "=" * (-len(payload_b64) % 4)
            payload_raw = urlsafe_b64decode(padded_b64.encode("utf-8")).decode("utf-8")
            payload_dict = json.loads(payload_raw)
            payload = ApprovalTokenPayload.model_validate(payload_dict)
        except Exception as e:
            return VerificationResult(
                is_valid=False,
                error_code="MALFORMED_PAYLOAD",
                error_message=f"Failed to decode token payload: {e}",
            )

        # 3. Check expiration
        now = int(datetime.now(UTC).timestamp())
        if now > payload.expires_at:
            logger.warning("Approval token expired", token_id=payload.token_id, expired_at=payload.expires_at)
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="TOKEN_EXPIRED",
                error_message="Approval token has expired",
            )

        # 4. Check run, tenant, and action match
        if payload.run_id != expected_run_id:
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="RUN_ID_MISMATCH",
                error_message=f"Token bound to run {payload.run_id}, not {expected_run_id}",
            )

        if payload.tenant_id != expected_tenant_id:
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="TENANT_MISMATCH",
                error_message=f"Token bound to tenant {payload.tenant_id}, not {expected_tenant_id}",
            )

        if payload.action != expected_action:
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="ACTION_MISMATCH",
                error_message=f"Token bound to action {payload.action}, not {expected_action}",
            )

        # 5. Check argument hash binding (Anti-Tampering)
        current_hash = hash_arguments(actual_arguments)
        if payload.payload_hash != current_hash:
            logger.warning(
                "Approval token argument hash mismatch: arguments were tampered with!",
                token_id=payload.token_id,
                expected_hash=payload.payload_hash,
                actual_hash=current_hash,
            )
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="ARGUMENTS_TAMPERED",
                error_message="Action arguments differ from those approved by the human operator",
            )

        # 6. Single-use check (Anti-Replay)
        if payload.token_id in self._consumed_token_ids:
            logger.warning("Replay attack detected: token already consumed", token_id=payload.token_id)
            return VerificationResult(
                is_valid=False,
                payload=payload,
                error_code="TOKEN_ALREADY_CONSUMED",
                error_message="Approval token has already been consumed and cannot be replayed",
            )

        # Mark token as consumed
        self._consumed_token_ids.add(payload.token_id)

        logger.info(
            "Approval token verified and consumed successfully",
            token_id=payload.token_id,
            run_id=expected_run_id,
            action=expected_action,
        )
        return VerificationResult(is_valid=True, payload=payload)

    def enforce_token_or_raise(
        self,
        token_str: str | None,
        expected_run_id: str,
        expected_tenant_id: str,
        expected_action: str,
        actual_arguments: dict[str, Any],
    ) -> ApprovalTokenPayload:
        """Convenience method that verifies token or raises a standard GovernanceError."""
        if not token_str:
            raise GovernanceError(
                message=f"Action '{expected_action}' requires a signed approval token",
                action=expected_action,
                details={"reason": "MISSING_APPROVAL_TOKEN", "run_id": expected_run_id},
            )

        result = self.verify_and_consume(
            token_str=token_str,
            expected_run_id=expected_run_id,
            expected_tenant_id=expected_tenant_id,
            expected_action=expected_action,
            actual_arguments=actual_arguments,
        )

        if not result.is_valid or result.payload is None:
            raise GovernanceError(
                message=f"Approval token verification failed: {result.error_message}",
                action=expected_action,
                details={
                    "error_code": result.error_code,
                    "run_id": expected_run_id,
                },
            )

        return result.payload
