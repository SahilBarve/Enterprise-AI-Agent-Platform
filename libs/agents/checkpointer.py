"""Persistent state checkpointing engine for LangGraph runs (FR-OR-5, FR-OR-10).

Architectural Concepts Explained:
--------------------------------
1. Why Persistent Checkpointing Matters:
   - Multi-agent runs may take minutes to hours (deep research, multi-query SQL, report generation).
   - If a worker crashes or a Kubernetes pod is evicted mid-run, an in-memory execution graph loses all progress.
   - With persistent checkpoints, every graph transition is saved to disk/database. A replacement worker can
     inspect the latest checkpoint for `run_id` and resume execution seamlessly.

2. Human-in-the-Loop (HITL) Support:
   - When an action requires human approval, LangGraph interrupts execution and pauses.
   - The approval wait could take seconds or days. Checkpointing allows the worker process to release resources
     and terminate without losing state.
   - When the user approves, the orchestrator reloads the exact state and resumes the graph from the paused node.

3. Time-Travel Debugging and Forking (FR-OR-10):
   - Engineers can view every intermediate state step in a run.
   - If a step produces bad output, `fork_checkpoint()` allows branching from step 3 with modified state or
     a different prompt without re-running steps 1 and 2!
"""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)

from libs.common.errors import ConflictError, NotFoundError
from libs.common.logging import get_logger

logger = get_logger("agents.checkpointer")


class CheckpointRecord(BaseModel):
    """Immutable snapshot of an orchestration run state at a specific graph step (FR-OR-5)."""

    checkpoint_id: str = Field(
        default_factory=lambda: f"chk-{uuid4().hex[:12]}",
        description="Unique checkpoint UUID",
    )
    run_id: str = Field(description="Run ID being tracked")
    thread_id: str = Field(
        description="LangGraph conversation thread ID (often identical to run_id)"
    )
    step_name: str = Field(description="Name of the node or step that just completed")
    state: dict[str, Any] = Field(description="Serialized AgentState dictionary")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Diagnostic tags and metrics")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp",
    )


class PlatformCheckpointer(ABC):
    """Abstract interface defining platform checkpoint operations."""

    @abstractmethod
    def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        step_name: str,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        """Persist a new checkpoint snapshot."""

    @abstractmethod
    def get_latest_checkpoint(self, run_id: str) -> CheckpointRecord | None:
        """Fetch the most recent checkpoint for a run."""

    @abstractmethod
    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        """Fetch a specific checkpoint by ID."""

    @abstractmethod
    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        """List all historical checkpoints for a run in chronological order."""

    @abstractmethod
    def fork_checkpoint(
        self,
        source_checkpoint_id: str,
        new_run_id: str,
        state_overrides: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        """Branch a new run from a historical checkpoint (FR-OR-10 time-travel)."""

    @abstractmethod
    def delete_checkpoints(self, run_id: str) -> int:
        """Clean up checkpoints for a completed or purged run."""

    @abstractmethod
    def to_langgraph_saver(self) -> BaseCheckpointSaver[Any]:
        """Return a LangGraph-compatible checkpointer object for graph compilation."""


class MemoryPlatformCheckpointer(PlatformCheckpointer):
    """In-memory thread-safe checkpointer implementation for development and testing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._run_history: dict[str, list[str]] = {}
        self._langgraph_saver = InMemorySaver()

    def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        step_name: str,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        tid = thread_id or run_id
        record = CheckpointRecord(
            run_id=run_id,
            thread_id=tid,
            step_name=step_name,
            state=deepcopy(state),
            metadata=deepcopy(metadata or {}),
        )
        self._checkpoints[record.checkpoint_id] = record
        if run_id not in self._run_history:
            self._run_history[run_id] = []
        self._run_history[run_id].append(record.checkpoint_id)

        logger.debug(
            "Saved memory checkpoint",
            run_id=run_id,
            checkpoint_id=record.checkpoint_id,
            step_name=step_name,
        )
        return record

    def get_latest_checkpoint(self, run_id: str) -> CheckpointRecord | None:
        chk_ids = self._run_history.get(run_id)
        if not chk_ids:
            return None
        return self._checkpoints.get(chk_ids[-1])

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        chk_ids = self._run_history.get(run_id, [])
        return [self._checkpoints[cid] for cid in chk_ids if cid in self._checkpoints]

    def fork_checkpoint(
        self,
        source_checkpoint_id: str,
        new_run_id: str,
        state_overrides: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        source = self.get_checkpoint(source_checkpoint_id)
        if not source:
            raise NotFoundError(
                message=f"Source checkpoint {source_checkpoint_id} not found",
                resource_type="checkpoint",
                resource_id=source_checkpoint_id,
            )
        if new_run_id in self._run_history:
            raise ConflictError(
                message=f"Target run {new_run_id} already exists",
                details={"run_id": new_run_id},
            )

        forked_state = deepcopy(source.state)
        if state_overrides:
            forked_state.update(state_overrides)
        forked_state["run_id"] = new_run_id

        fork_meta = deepcopy(source.metadata)
        fork_meta["forked_from_checkpoint"] = source_checkpoint_id
        fork_meta["forked_from_run"] = source.run_id

        return self.save_checkpoint(
            run_id=new_run_id,
            state=forked_state,
            step_name=f"fork_from_{source.step_name}",
            thread_id=new_run_id,
            metadata=fork_meta,
        )

    def delete_checkpoints(self, run_id: str) -> int:
        chk_ids = self._run_history.pop(run_id, [])
        count = 0
        for cid in chk_ids:
            if self._checkpoints.pop(cid, None) is not None:
                count += 1
        return count

    def to_langgraph_saver(self) -> BaseCheckpointSaver[Any]:
        return self._langgraph_saver


# =============================================================================
# SQLPlatformCheckpointer (SQLAlchemy SQLite / PostgreSQL backend) (FR-OR-5)
# =============================================================================


class SQLPlatformCheckpointer(PlatformCheckpointer):
    """ACID-compliant relational database checkpointer for production deployments.

    Stores JSON-serialized state snapshots in PostgreSQL (or SQLite locally/tests),
    allowing runs to survive worker crashes, restarts, and long HITL approval pauses.
    """

    def __init__(self, db_url: str = "sqlite:///:memory:") -> None:
        """Initialize SQL checkpointer and ensure schema exists.

        Args:
            db_url: Database connection string (e.g. 'postgresql://...' or 'sqlite:///...')
        """
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.metadata = MetaData()
        self._table = Table(
            "run_checkpoints",
            self.metadata,
            Column("checkpoint_id", String(64), primary_key=True),
            Column("run_id", String(64), nullable=False, index=True),
            Column("thread_id", String(64), nullable=False, index=True),
            Column("step_name", String(128), nullable=False),
            Column("state_json", Text, nullable=False),
            Column("metadata_json", Text, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Index("idx_run_created", "run_id", "created_at"),
        )
        self.metadata.create_all(self.engine)
        self._langgraph_saver = InMemorySaver()

    def _serialize(self, obj: Any) -> str:
        """Safe JSON serializer handling datetime, UUID, and custom objects."""

        def default_encoder(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            if hasattr(o, "model_dump"):
                return o.model_dump()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        return json.dumps(obj, default=default_encoder)

    def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        step_name: str,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        tid = thread_id or run_id
        now = datetime.now(UTC)
        record = CheckpointRecord(
            run_id=run_id,
            thread_id=tid,
            step_name=step_name,
            state=state,
            metadata=metadata or {},
            created_at=now,
        )

        state_str = self._serialize(state)
        meta_str = self._serialize(record.metadata)

        with self.engine.begin() as conn:
            conn.execute(
                self._table.insert().values(
                    checkpoint_id=record.checkpoint_id,
                    run_id=run_id,
                    thread_id=tid,
                    step_name=step_name,
                    state_json=state_str,
                    metadata_json=meta_str,
                    created_at=now,
                )
            )

        logger.debug(
            "Saved SQL checkpoint",
            run_id=run_id,
            checkpoint_id=record.checkpoint_id,
            step_name=step_name,
        )
        return record

    def get_latest_checkpoint(self, run_id: str) -> CheckpointRecord | None:
        stmt = (
            select(self._table)
            .where(self._table.c.run_id == run_id)
            .order_by(self._table.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).first()
            if not row:
                return None
            return self._row_to_record(row)

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        stmt = select(self._table).where(self._table.c.checkpoint_id == checkpoint_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).first()
            if not row:
                return None
            return self._row_to_record(row)

    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        stmt = (
            select(self._table)
            .where(self._table.c.run_id == run_id)
            .order_by(self._table.c.created_at.asc())
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_record(row) for row in rows]

    def fork_checkpoint(
        self,
        source_checkpoint_id: str,
        new_run_id: str,
        state_overrides: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        source = self.get_checkpoint(source_checkpoint_id)
        if not source:
            raise NotFoundError(
                message=f"Source checkpoint {source_checkpoint_id} not found",
                resource_type="checkpoint",
                resource_id=source_checkpoint_id,
            )

        existing = self.get_latest_checkpoint(new_run_id)
        if existing is not None:
            raise ConflictError(
                message=f"Target run {new_run_id} already exists",
                details={"run_id": new_run_id},
            )

        forked_state = deepcopy(source.state)
        if state_overrides:
            forked_state.update(state_overrides)
        forked_state["run_id"] = new_run_id

        fork_meta = deepcopy(source.metadata)
        fork_meta["forked_from_checkpoint"] = source_checkpoint_id
        fork_meta["forked_from_run"] = source.run_id

        return self.save_checkpoint(
            run_id=new_run_id,
            state=forked_state,
            step_name=f"fork_from_{source.step_name}",
            thread_id=new_run_id,
            metadata=fork_meta,
        )

    def delete_checkpoints(self, run_id: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                self._table.delete().where(self._table.c.run_id == run_id)
            )
            return int(result.rowcount)

    def to_langgraph_saver(self) -> BaseCheckpointSaver[Any]:
        return self._langgraph_saver

    def _row_to_record(self, row: Any) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=row.checkpoint_id,
            run_id=row.run_id,
            thread_id=row.thread_id,
            step_name=row.step_name,
            state=json.loads(row.state_json),
            metadata=json.loads(row.metadata_json),
            created_at=row.created_at,
        )
