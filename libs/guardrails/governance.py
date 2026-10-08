"""Risk-tiered policy engine and governance gate (FR-OR-6, FR-SEC-3 to FR-SEC-6, P0 Pillar).

Architectural Concepts Explained:
--------------------------------
1. Why Policy-Driven Risk Tiering?
   - In an enterprise multi-agent platform, not all actions have equal blast radiuses.
   - Reading documentation (`retrieve_docs`) or searching the web (`search_web`) is safe to execute autonomously.
   - Writing to an operational database (`run_sql_write`), sending webhooks to third-party endpoints,
     or modifying infrastructure has permanent side effects.
   - Hardcoding checks inside agents creates maintenance nightmares and bypass risks.
   - The GovernanceGate acts as a centralized security checkpoint through which all agent actions must pass.

2. Four-Tier Security Taxonomy:
   - Tier 0 (Read-Only): Auto-approved without pausing.
   - Tier 1 (Low-Risk): Auto-approved with structured audit logging.
   - Tier 2 (High-Risk Write): Execution halted; requires human-in-the-loop sign-off with HMAC token.
   - Tier 3 (Prohibited): Strictly forbidden actions; blocked unconditionally.
"""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

from libs.common.errors import GovernanceError
from libs.common.logging import get_logger
from libs.guardrails.tokens import ApprovalTokenManager

logger = get_logger("guardrails.governance")


class ActionRiskTier(IntEnum):
    """Categorical risk classification for agent actions and tool calls."""

    TIER_0_READ_ONLY = 0
    TIER_1_LOW_RISK = 1
    TIER_2_HIGH_RISK = 2
    TIER_3_PROHIBITED = 3


class PolicyDecision(BaseModel):
    """Decision emitted by the Governance Gate evaluating a proposed action."""

    allowed: bool
    risk_tier: ActionRiskTier
    requires_approval: bool
    token_verified: bool = False
    action: str
    reason: str
    remediation_hint: str | None = None


# Default platform-wide action to risk tier mappings
DEFAULT_ACTION_TIERS: dict[str, ActionRiskTier] = {
    # Tier 0: Read-Only Actions
    "retrieve_docs": ActionRiskTier.TIER_0_READ_ONLY,
    "search_web": ActionRiskTier.TIER_0_READ_ONLY,
    "run_sql_read": ActionRiskTier.TIER_0_READ_ONLY,
    "profile_data": ActionRiskTier.TIER_0_READ_ONLY,
    "preview_report": ActionRiskTier.TIER_0_READ_ONLY,
    # Tier 1: Low-Risk Operations
    "execute_code_sandbox": ActionRiskTier.TIER_1_LOW_RISK,
    "transform_dataset": ActionRiskTier.TIER_1_LOW_RISK,
    "draft_report": ActionRiskTier.TIER_1_LOW_RISK,
    "generate_report": ActionRiskTier.TIER_1_LOW_RISK,
    # Tier 2: High-Risk Mutations (Mandatory Token Required)
    "run_sql_write": ActionRiskTier.TIER_2_HIGH_RISK,
    "execute_code": ActionRiskTier.TIER_2_HIGH_RISK,
    "send_webhook": ActionRiskTier.TIER_2_HIGH_RISK,
    "send_email": ActionRiskTier.TIER_2_HIGH_RISK,
    "publish_report": ActionRiskTier.TIER_2_HIGH_RISK,
    "modify_config": ActionRiskTier.TIER_2_HIGH_RISK,
    # Tier 3: Prohibited Actions (Unconditionally Blocked)
    "shell_exec": ActionRiskTier.TIER_3_PROHIBITED,
    "raw_bash": ActionRiskTier.TIER_3_PROHIBITED,
    "drop_database": ActionRiskTier.TIER_3_PROHIBITED,
    "read_secrets": ActionRiskTier.TIER_3_PROHIBITED,
    "exfiltrate_data": ActionRiskTier.TIER_3_PROHIBITED,
}


class GovernancePolicy(BaseModel):
    """Configurable governance policy with per-tenant overrides (FR-OR-8, FR-SEC-5)."""

    tenant_id: str
    custom_tiers: dict[str, ActionRiskTier] = Field(default_factory=dict)
    require_approval_for_all_code: bool = Field(default=False)
    allow_sql_writes: bool = Field(default=True)

    def get_action_tier(self, action: str) -> ActionRiskTier:
        """Resolve the effective risk tier for an action under this policy."""
        # 1. Custom tenant override
        if action in self.custom_tiers:
            return self.custom_tiers[action]

        # 2. Strict policy toggles
        if action == "execute_code_sandbox" and self.require_approval_for_all_code:
            return ActionRiskTier.TIER_2_HIGH_RISK
        if action == "run_sql_write" and not self.allow_sql_writes:
            return ActionRiskTier.TIER_3_PROHIBITED

        # 3. Default platform mapping (defaulting unknown actions to Tier 2 for safety)
        return DEFAULT_ACTION_TIERS.get(action, ActionRiskTier.TIER_2_HIGH_RISK)


class GovernanceGate:
    """Centralized governance policy gate enforcing risk-tier checks and approval tokens."""

    def __init__(
        self,
        token_manager: ApprovalTokenManager | None = None,
        tenant_policies: dict[str, GovernancePolicy] | None = None,
    ) -> None:
        """Initialize Governance Gate.

        Args:
            token_manager: Manager for validating cryptographic approval tokens.
            tenant_policies: Dictionary of custom tenant policies.
        """
        self.token_manager = token_manager or ApprovalTokenManager()
        self.tenant_policies = tenant_policies or {}

    def get_policy(self, tenant_id: str) -> GovernancePolicy:
        """Retrieve tenant policy or default."""
        return self.tenant_policies.get(tenant_id, GovernancePolicy(tenant_id=tenant_id))

    def evaluate(
        self,
        run_id: str,
        tenant_id: str,
        action: str,
        arguments: dict[str, Any],
        approval_token: str | None = None,
    ) -> PolicyDecision:
        """Evaluate action against policy rules and verify approval token if Tier 2.

        Args:
            run_id: Active run ID.
            tenant_id: Tenant scope.
            action: Action identifier.
            arguments: Exact execution parameters.
            approval_token: Optional HMAC signed approval token.

        Returns:
            PolicyDecision detailing whether the action is permitted.
        """
        policy = self.get_policy(tenant_id)
        tier = policy.get_action_tier(action)

        # -------------------------------------------------------------
        # Tier 3: Prohibited Actions
        # -------------------------------------------------------------
        if tier == ActionRiskTier.TIER_3_PROHIBITED:
            logger.warning(
                "Governance gate BLOCKED prohibited Tier 3 action",
                run_id=run_id,
                tenant_id=tenant_id,
                action=action,
            )
            return PolicyDecision(
                allowed=False,
                risk_tier=tier,
                requires_approval=False,
                token_verified=False,
                action=action,
                reason=f"Action '{action}' is strictly prohibited by security policy (Tier 3).",
                remediation_hint="Use standard platform tools instead of prohibited system actions.",
            )

        # -------------------------------------------------------------
        # Tier 0 & Tier 1: Auto-Approved Actions
        # -------------------------------------------------------------
        if tier in (ActionRiskTier.TIER_0_READ_ONLY, ActionRiskTier.TIER_1_LOW_RISK):
            logger.info(
                "Governance gate auto-approved action",
                run_id=run_id,
                tenant_id=tenant_id,
                action=action,
                risk_tier=tier.name,
            )
            return PolicyDecision(
                allowed=True,
                risk_tier=tier,
                requires_approval=False,
                token_verified=False,
                action=action,
                reason=f"Action '{action}' is authorized without approval ({tier.name}).",
            )

        # -------------------------------------------------------------
        # Tier 2: High-Risk Mutations (Requires Valid Approval Token)
        # -------------------------------------------------------------
        if not approval_token:
            logger.info(
                "Governance gate requires human approval for Tier 2 action",
                run_id=run_id,
                tenant_id=tenant_id,
                action=action,
            )
            return PolicyDecision(
                allowed=False,
                risk_tier=tier,
                requires_approval=True,
                token_verified=False,
                action=action,
                reason=f"Action '{action}' is a high-risk operation requiring human approval token (Tier 2).",
                remediation_hint="Request operator approval; provide issued HMAC approval token to proceed.",
            )

        # Verify cryptographic approval token
        verification = self.token_manager.verify_and_consume(
            token_str=approval_token,
            expected_run_id=run_id,
            expected_tenant_id=tenant_id,
            expected_action=action,
            actual_arguments=arguments,
        )

        if not verification.is_valid:
            logger.warning(
                "Governance gate rejected invalid approval token",
                run_id=run_id,
                tenant_id=tenant_id,
                action=action,
                error_code=verification.error_code,
            )
            return PolicyDecision(
                allowed=False,
                risk_tier=tier,
                requires_approval=True,
                token_verified=False,
                action=action,
                reason=f"Approval token invalid: {verification.error_message} ({verification.error_code})",
                remediation_hint="Request a new approval token with matching run ID and arguments.",
            )

        logger.info(
            "Governance gate validated Tier 2 action via approval token",
            run_id=run_id,
            tenant_id=tenant_id,
            action=action,
            token_id=verification.payload.token_id if verification.payload else None,
        )
        return PolicyDecision(
            allowed=True,
            risk_tier=tier,
            requires_approval=True,
            token_verified=True,
            action=action,
            reason=f"Action '{action}' successfully authorized via cryptographic approval token.",
        )

    def enforce_or_raise(
        self,
        run_id: str,
        tenant_id: str,
        action: str,
        arguments: dict[str, Any],
        approval_token: str | None = None,
    ) -> PolicyDecision:
        """Evaluate action and raise GovernanceError if not permitted."""
        decision = self.evaluate(
            run_id=run_id,
            tenant_id=tenant_id,
            action=action,
            arguments=arguments,
            approval_token=approval_token,
        )
        if not decision.allowed:
            raise GovernanceError(
                message=decision.reason,
                action=action,
                details={
                    "risk_tier": decision.risk_tier,
                    "requires_approval": decision.requires_approval,
                    "run_id": run_id,
                },
            )
        return decision
