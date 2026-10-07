"""Security, guardrails, and governance gate package (FR-OR-6, FR-SEC-3 to FR-SEC-6).

This package provides:
- HMAC-SHA256 signed single-use approval tokens cryptographically bound to exact action arguments.
- Risk-tiered policy engine classifying actions into Tier 0 (Read-Only), Tier 1 (Low-Risk),
  Tier 2 (High-Risk Write requiring token), and Tier 3 (Prohibited).
- Governance gate enforcing policy compliance across supervisor steps and MCP tools.
"""

from libs.guardrails.governance import (
    ActionRiskTier,
    GovernanceGate,
    GovernancePolicy,
    PolicyDecision,
)
from libs.guardrails.tokens import (
    ApprovalTokenManager,
    ApprovalTokenPayload,
    VerificationResult,
)

__all__ = [
    "ActionRiskTier",
    "ApprovalTokenManager",
    "ApprovalTokenPayload",
    "GovernanceGate",
    "GovernancePolicy",
    "PolicyDecision",
    "VerificationResult",
]
