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
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
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
# SQLCheckpointSaver (Durable LangGraph CheckpointSaver) (FR-OR-5)
# =============================================================================


class SQLCheckpointSaver(BaseCheckpointSaver[str]):
    """Thread-safe, durable relational database checkpoint saver for LangGraph (FR-OR-5).

    Persists graph step snapshots, delta channel versions, and pending task writes into
    relational tables (SQLite / PostgreSQL), enabling seamless multi-step resume across
    worker crashes and human-in-the-loop pauses.
    """

    def __init__(self, engine: Any) -> None:
        """Initialize schema tables and synchronization primitives.

        Args:
            engine: SQLAlchemy database engine.
        """
        super().__init__()
        self.engine = engine
        self.lock = threading.Lock()
        self.metadata = MetaData()

        self.t_checkpoints = Table(
            "lg_checkpoints",
            self.metadata,
            Column("thread_id", String(128), primary_key=True),
            Column("checkpoint_ns", String(128), primary_key=True),
            Column("checkpoint_id", String(128), primary_key=True),
            Column("parent_checkpoint_id", String(128), nullable=True),
            Column("cp_type", String(64), nullable=False),
            Column("cp_data", LargeBinary, nullable=False),
            Column("meta_type", String(64), nullable=False),
            Column("meta_data", LargeBinary, nullable=False),
        )

        self.t_blobs = Table(
            "lg_blobs",
            self.metadata,
            Column("thread_id", String(128), primary_key=True),
            Column("checkpoint_ns", String(128), primary_key=True),
            Column("channel", String(128), primary_key=True),
            Column("version", String(128), primary_key=True),
            Column("blob_type", String(64), nullable=False),
            Column("blob_data", LargeBinary, nullable=False),
        )

        self.t_writes = Table(
            "lg_writes",
            self.metadata,
            Column("thread_id", String(128), primary_key=True),
            Column("checkpoint_ns", String(128), primary_key=True),
            Column("checkpoint_id", String(128), primary_key=True),
            Column("task_id", String(128), primary_key=True),
            Column("idx", Integer, primary_key=True),
            Column("channel", String(128), nullable=False),
            Column("write_type", String(64), nullable=False),
            Column("write_data", LargeBinary, nullable=False),
            Column("task_path", String(256), nullable=False, default=""),
        )
        self.metadata.create_all(self.engine)

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Load and deserialize channel blobs for a specific checkpoint version."""
        result: dict[str, Any] = {}
        with self.engine.connect() as conn:
            for k, ver in versions.items():
                s = select(self.t_blobs.c.blob_type, self.t_blobs.c.blob_data).where(
                    self.t_blobs.c.thread_id == thread_id,
                    self.t_blobs.c.checkpoint_ns == checkpoint_ns,
                    self.t_blobs.c.channel == k,
                    self.t_blobs.c.version == str(ver),
                )
                row = conn.execute(s).fetchone()
                if row:
                    b_type, b_data = row
                    if b_type != "empty":
                        result[k] = self.serde.loads_typed((b_type, b_data))
        return result

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch checkpoint tuple for given thread_id and optional checkpoint_id."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.engine.connect() as conn:
            if checkpoint_id:
                s = select(self.t_checkpoints).where(
                    self.t_checkpoints.c.thread_id == thread_id,
                    self.t_checkpoints.c.checkpoint_ns == checkpoint_ns,
                    self.t_checkpoints.c.checkpoint_id == checkpoint_id,
                )
                row = conn.execute(s).fetchone()
            else:
                s = select(self.t_checkpoints).where(
                    self.t_checkpoints.c.thread_id == thread_id,
                    self.t_checkpoints.c.checkpoint_ns == checkpoint_ns,
                ).order_by(self.t_checkpoints.c.checkpoint_id.desc()).limit(1)
                row = conn.execute(s).fetchone()

            if not row:
                return None

            t_id, ns, c_id, parent_id, cp_t, cp_d, m_t, m_d = row
            cp_dict: Checkpoint = self.serde.loads_typed((cp_t, cp_d))
            meta_dict = self.serde.loads_typed((m_t, m_d))

            sw = select(self.t_writes).where(
                self.t_writes.c.thread_id == thread_id,
                self.t_writes.c.checkpoint_ns == checkpoint_ns,
                self.t_writes.c.checkpoint_id == c_id,
            )
            w_rows = conn.execute(sw).fetchall()
            pending_writes = [
                (w[3], w[5], self.serde.loads_typed((w[6], w[7]))) for w in w_rows
            ]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": c_id,
                    }
                },
                checkpoint={
                    **cp_dict,
                    "channel_values": self._load_blobs(
                        thread_id, checkpoint_ns, cp_dict["channel_versions"]
                    ),
                },
                metadata=meta_dict,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    }
                    if parent_id
                    else None
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a new checkpoint snapshot and channel versions to the database."""
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        cp_type, cp_data = self.serde.dumps_typed(c)
        meta_type, meta_data = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        parent_id = config["configurable"].get("checkpoint_id")

        with self.lock, self.engine.begin() as conn:
            for k, v in new_versions.items():
                if k in values:
                    b_type, b_data = self.serde.dumps_typed(values[k])
                else:
                    b_type, b_data = ("empty", b"")

                conn.execute(
                    delete(self.t_blobs).where(
                        self.t_blobs.c.thread_id == thread_id,
                        self.t_blobs.c.checkpoint_ns == checkpoint_ns,
                        self.t_blobs.c.channel == k,
                        self.t_blobs.c.version == str(v),
                    )
                )
                conn.execute(
                    self.t_blobs.insert().values(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        channel=k,
                        version=str(v),
                        blob_type=b_type,
                        blob_data=b_data,
                    )
                )

            conn.execute(
                delete(self.t_checkpoints).where(
                    self.t_checkpoints.c.thread_id == thread_id,
                    self.t_checkpoints.c.checkpoint_ns == checkpoint_ns,
                    self.t_checkpoints.c.checkpoint_id == checkpoint["id"],
                )
            )
            conn.execute(
                self.t_checkpoints.insert().values(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint["id"],
                    parent_checkpoint_id=parent_id,
                    cp_type=cp_type,
                    cp_data=cp_data,
                    meta_type=meta_type,
                    meta_data=meta_data,
                )
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending node writes associated with an interrupted or completed step."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self.lock, self.engine.begin() as conn:
            for idx, (c, v) in enumerate(writes):
                idx_val = WRITES_IDX_MAP.get(c, idx)
                w_type, w_data = self.serde.dumps_typed(v)

                conn.execute(
                    delete(self.t_writes).where(
                        self.t_writes.c.thread_id == thread_id,
                        self.t_writes.c.checkpoint_ns == checkpoint_ns,
                        self.t_writes.c.checkpoint_id == checkpoint_id,
                        self.t_writes.c.task_id == task_id,
                        self.t_writes.c.idx == idx_val,
                    )
                )
                conn.execute(
                    self.t_writes.insert().values(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx_val,
                        channel=c,
                        write_type=w_type,
                        write_data=w_data,
                        task_path=task_path,
                    )
                )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: ARG002
        before: RunnableConfig | None = None,  # noqa: ARG002
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List historical checkpoint tuples for time-travel inspection."""
        if not config:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self.engine.connect() as conn:
            s = select(self.t_checkpoints).where(
                self.t_checkpoints.c.thread_id == thread_id,
                self.t_checkpoints.c.checkpoint_ns == checkpoint_ns,
            ).order_by(self.t_checkpoints.c.checkpoint_id.desc())
            rows = conn.execute(s).fetchall()
            if limit is not None and limit > 0:
                rows = rows[:limit]

            for r in rows:
                c_id = r[2]
                tup = self.get_tuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": c_id,
                        }
                    }
                )
                if tup:
                    yield tup

    def delete_thread(self, thread_id: str) -> None:
        """Purge all checkpoints, blobs, and writes for a given thread ID."""
        with self.lock, self.engine.begin() as conn:
            conn.execute(
                delete(self.t_checkpoints).where(
                    self.t_checkpoints.c.thread_id == thread_id
                )
            )
            conn.execute(
                delete(self.t_blobs).where(self.t_blobs.c.thread_id == thread_id)
            )
            conn.execute(
                delete(self.t_writes).where(self.t_writes.c.thread_id == thread_id)
            )

    def get_next_version(self, current: str | None, channel: None = None) -> str:  # noqa: ARG002
        """Calculate next monotonic channel version identifier."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)


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
        self._langgraph_saver = SQLCheckpointSaver(self.engine)

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
