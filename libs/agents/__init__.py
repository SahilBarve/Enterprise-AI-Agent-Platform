"""Agents package for multi-agent supervisor and subgraphs (FR-OR-1 to FR-OR-12).

This package provides:
- State definitions, reducers, and schemas for LangGraph multi-agent execution.
- Persistent checkpointing backends (in-memory, SQL/PostgreSQL) with time-travel and crash-resilience.
- Supervisor graph with Planner, Executor, Critic, and Budget Guard nodes.
- Specialized subgraphs for Document RAG, Web Research, SQL Analytics, Data Processing, and Reports.
"""

from libs.agents.state import (
    AgentState,
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
)

__all__ = [
    "AgentState",
    "AgentType",
    "ApprovalRequest",
    "BudgetLimits",
    "BudgetUsage",
    "Plan",
    "PlanStep",
    "RunStatus",
    "StepActionType",
    "StepEvent",
    "append_artifacts",
    "append_events",
]
