"""API Gateway router for multi-agent runs, HITL approvals, and time-travel debugging (FR-OR-1, FR-OR-6, FR-OR-9, FR-OR-10).

Architectural Concepts Explained:
--------------------------------
1. Orchestrator API Gateway (FR-OR-1):
   - External clients (frontend web UI, CLI, webhooks) interact with the multi-agent system via
     standard REST endpoints under `/api/v1/runs`.
   - Long-running multi-agent runs can be monitored, paused, resumed, or forked.

2. Human-in-the-Loop Sign-off (`POST /runs/{id}/approve`):
   - When a run pauses with `status: awaiting_approval`, operators review the pending action and exact parameters.
   - Submitting an approval creates an HMAC-SHA256 token cryptographically bound to those exact parameters.
   - The router resumes the LangGraph supervisor using the token.

3. Time-Travel Debugging & Checkpoint Inspection (`POST /runs/{id}/fork`, `GET /runs/{id}/checkpoints`):
   - Engineers can inspect the chronological evolution of run state across checkpoints.
   - The `/fork` endpoint branches a new run from any historical step with modified parameters.
"""

import json
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from libs.agents.checkpointer import CheckpointRecord, PlatformCheckpointer
from libs.agents.state import (
    AgentState,
    ApprovalRequest,
    BudgetLimits,
    BudgetUsage,
    Plan,
    RunStatus,
)
from libs.agents.supervisor import MultiAgentSupervisor
from libs.common.errors import NotFoundError, ValidationError
from libs.common.logging import get_logger
from libs.guardrails.tokens import ApprovalTokenManager
from services.gateway.dependencies import (
    get_checkpointer,
    get_supervisor,
    get_token_manager,
)

logger = get_logger("gateway.orchestrator")

router = APIRouter(prefix="/runs", tags=["Agent Orchestration & Runs"])


# =============================================================================
# Request & Response Schemas
# =============================================================================


class CreateRunRequest(BaseModel):
    """Payload to initiate a new multi-agent run."""

    goal: str = Field(description="Natural language objective for the multi-agent supervisor")
    tenant_id: str = Field(description="Tenant scope for multi-tenant isolation")
    user_id: str = Field(default="user-default", description="Requesting user ID")
    max_steps: int = Field(default=25, ge=1, le=100, description="Step budget limit")
    max_tokens: int = Field(default=50000, ge=100, description="Token budget limit")
    max_cost_usd: float = Field(default=2.00, ge=0.01, description="Monetary budget limit in USD")
    max_wall_clock_seconds: float = Field(default=300.0, ge=5.0, description="Timeout in seconds")


class RunResponse(BaseModel):
    """Standardized representation of an orchestration run state."""

    run_id: str
    tenant_id: str
    user_id: str
    status: RunStatus
    goal: str
    plan: Plan | None = None
    current_step_id: str | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    budget_usage: BudgetUsage | None = None
    pending_approval: ApprovalRequest | None = None
    error: str | None = None
    final_response: str | None = None


class SubmitApprovalRequest(BaseModel):
    """Operator decision payload for an interrupted run awaiting sign-off."""

    action: str = Field(description="Action being authorized")
    arguments: dict[str, Any] = Field(description="Exact arguments approved by operator")
    approved: bool = Field(description="True=approve and resume, False=reject")
    approver_id: str = Field(default="admin-operator", description="Identifier of human approver")
    notes: str | None = Field(default=None, description="Optional sign-off feedback")


class ForkRunRequest(BaseModel):
    """Payload to branch a new run from a historical checkpoint (time-travel debugging)."""

    source_checkpoint_id: str = Field(description="Existing checkpoint to fork from")
    new_run_id: str = Field(description="New target run ID")
    state_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="State modifications to apply at the branch point",
    )


def _state_to_response(state: AgentState) -> RunResponse:
    """Helper to convert AgentState dictionary into a structured RunResponse model."""
    pending_appr: ApprovalRequest | None = None
    paused_dict = cast(dict[str, Any], state)
    if "__interrupt__" in paused_dict and paused_dict["__interrupt__"]:
        interrupt_item = paused_dict["__interrupt__"][0]
        interrupt_val = interrupt_item.value if hasattr(interrupt_item, "value") else interrupt_item
        if isinstance(interrupt_val, dict):
            if "value" in interrupt_val and isinstance(interrupt_val["value"], dict):
                interrupt_val = interrupt_val["value"]
            pending_appr = ApprovalRequest.model_validate(interrupt_val)

    return RunResponse(
        run_id=state.get("run_id", "unknown"),
        tenant_id=state.get("tenant_id", "unknown"),
        user_id=state.get("user_id", "unknown"),
        status=state.get("status", RunStatus.RUNNING),
        goal=state.get("goal", ""),
        plan=state.get("plan"),
        current_step_id=state.get("current_step_id"),
        step_results=state.get("step_results", {}),
        budget_usage=state.get("budget_usage"),
        pending_approval=pending_appr,
        error=state.get("error"),
        final_response=state.get("final_response"),
    )


# =============================================================================
# Run Lifecycle Endpoints (FR-OR-1, FR-OR-6, FR-OR-9, FR-OR-10)
# =============================================================================


@router.post(
    "",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start new multi-agent run",
    description="Decomposes user goal, initiates supervisor execution, and returns run status.",
)
def create_run(
    request: CreateRunRequest,
    supervisor: Annotated[MultiAgentSupervisor, Depends(get_supervisor)],
) -> RunResponse:
    """Create and start a new orchestration run."""
    limits = BudgetLimits(
        max_steps=request.max_steps,
        max_tokens=request.max_tokens,
        max_cost_usd=request.max_cost_usd,
        max_wall_clock_seconds=request.max_wall_clock_seconds,
    )

    state = supervisor.start_run(
        goal=request.goal,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        budget_limits=limits,
    )
    return _state_to_response(state)


@router.get(
    "/{run_id}",
    response_model=RunResponse,
    summary="Get run status and state snapshot",
    description="Inspect current execution state, plan progress, budget usage, and results.",
)
def get_run(
    run_id: str,
    supervisor: Annotated[MultiAgentSupervisor, Depends(get_supervisor)],
) -> RunResponse:
    """Fetch run status by ID."""
    state = supervisor.get_state(run_id)
    if not state:
        raise NotFoundError(f"Run {run_id} not found", resource_type="run", resource_id=run_id)
    return _state_to_response(state)


@router.get(
    "/{run_id}/events",
    summary="Get or stream run timeline events (FR-GW-6)",
    description="Retrieve all timeline events, or stream real-time events over Server-Sent Events (SSE).",
)
def get_run_events(
    run_id: str,
    supervisor: Annotated[MultiAgentSupervisor, Depends(get_supervisor)],
    stream: bool = Query(default=False, description="Stream events as Server-Sent Events (SSE)"),
) -> Any:
    """Retrieve or stream chronological run timeline events."""
    state = supervisor.get_state(run_id)
    if not state:
        raise NotFoundError(f"Run {run_id} not found", resource_type="run", resource_id=run_id)

    events = state.get("events", [])

    if not stream:
        return [e.model_dump() if hasattr(e, "model_dump") else e for e in events]

    # Server-Sent Events (SSE) streaming formatter
    def event_generator() -> Any:
        for event in events:
            raw = (
                event.model_dump_json()
                if hasattr(event, "model_dump_json")
                else json.dumps(event, default=str)
            )
            yield f"data: {raw}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/{run_id}/approve",
    response_model=RunResponse,
    summary="Submit HITL decision for paused run (FR-OR-6, P0 Pillar)",
    description="Approves or rejects a high-risk Tier 2 action; generates signed HMAC token and resumes graph.",
)
def approve_run(
    run_id: str,
    decision: SubmitApprovalRequest,
    supervisor: Annotated[MultiAgentSupervisor, Depends(get_supervisor)],
    token_manager: Annotated[ApprovalTokenManager, Depends(get_token_manager)],
) -> RunResponse:
    """Approve or reject an interrupted run."""
    state = supervisor.get_state(run_id)
    if not state:
        raise NotFoundError(f"Run {run_id} not found", resource_type="run", resource_id=run_id)

    tenant_id = state.get("tenant_id", "tenant-default")

    if not decision.approved:
        logger.info("Human operator rejected run approval", run_id=run_id, action=decision.action)
        # Cancel run
        state["status"] = RunStatus.CANCELLED
        state["error"] = f"Action '{decision.action}' was rejected by operator ({decision.approver_id})"
        supervisor.checkpointer.save_checkpoint(
            run_id=run_id,
            state=dict(state),
            step_name="approval_rejected",
        )
        return _state_to_response(state)

    # Operator approved: Issue HMAC-SHA256 signed single-use token bound to exact arguments
    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action=decision.action,
        arguments=decision.arguments,
    )

    logger.info("Resuming run with operator-issued approval token", run_id=run_id, action=decision.action)
    resumed_state = supervisor.resume_run(run_id=run_id, approval_token=token)
    return _state_to_response(resumed_state)


@router.get(
    "/{run_id}/checkpoints",
    response_model=list[CheckpointRecord],
    summary="List historical checkpoints (FR-OR-5)",
    description="Returns all persisted checkpoint records for audit and time-travel replay.",
)
def list_run_checkpoints(
    run_id: str,
    checkpointer: Annotated[PlatformCheckpointer, Depends(get_checkpointer)],
) -> list[CheckpointRecord]:
    """List checkpoints for a run."""
    return checkpointer.list_checkpoints(run_id)


@router.post(
    "/{run_id}/fork",
    response_model=CheckpointRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Fork run from checkpoint (FR-OR-10 Time-Travel Debugging)",
    description="Branches a new execution run from any historical checkpoint snapshot.",
)
def fork_run(
    run_id: str,
    request: ForkRunRequest,
    checkpointer: Annotated[PlatformCheckpointer, Depends(get_checkpointer)],
) -> CheckpointRecord:
    """Fork a historical checkpoint into a new run."""
    if not request.new_run_id:
        raise ValidationError("new_run_id is required")

    source_chk = checkpointer.get_checkpoint(request.source_checkpoint_id)
    if not source_chk or source_chk.run_id != run_id:
        raise NotFoundError(
            f"Source checkpoint {request.source_checkpoint_id} not found in run {run_id}",
            resource_type="checkpoint",
            resource_id=request.source_checkpoint_id,
        )

    return checkpointer.fork_checkpoint(
        source_checkpoint_id=request.source_checkpoint_id,
        new_run_id=request.new_run_id,
        state_overrides=request.state_overrides,
    )
