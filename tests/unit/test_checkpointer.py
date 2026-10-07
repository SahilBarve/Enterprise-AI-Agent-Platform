"""Unit tests for persistent checkpointers and crash recovery (FR-OR-5, FR-OR-10)."""

import pytest

from libs.agents.checkpointer import (
    MemoryPlatformCheckpointer,
    SQLPlatformCheckpointer,
)
from libs.common.errors import ConflictError, NotFoundError


@pytest.mark.unit
def test_memory_checkpointer_crud() -> None:
    """Validate in-memory checkpointer save, fetch, list, and delete."""
    cp = MemoryPlatformCheckpointer()
    run_id = "run-mem-01"

    state_step1 = {"run_id": run_id, "step": "planner", "plan": ["step1", "step2"]}
    rec1 = cp.save_checkpoint(run_id, state_step1, step_name="planner")
    assert rec1.checkpoint_id.startswith("chk-")
    assert rec1.step_name == "planner"
    assert rec1.state["step"] == "planner"

    state_step2 = {"run_id": run_id, "step": "executor", "result": "done"}
    rec2 = cp.save_checkpoint(run_id, state_step2, step_name="executor")

    latest = cp.get_latest_checkpoint(run_id)
    assert latest is not None
    assert latest.checkpoint_id == rec2.checkpoint_id
    assert latest.step_name == "executor"

    history = cp.list_checkpoints(run_id)
    assert len(history) == 2
    assert history[0].step_name == "planner"
    assert history[1].step_name == "executor"

    # Fetch by ID
    fetched = cp.get_checkpoint(rec1.checkpoint_id)
    assert fetched is not None
    assert fetched.step_name == "planner"

    # Cleanup
    deleted = cp.delete_checkpoints(run_id)
    assert deleted == 2
    assert cp.get_latest_checkpoint(run_id) is None


@pytest.mark.unit
def test_memory_checkpointer_time_travel_forking() -> None:
    """Validate branching a new run from a historical checkpoint (FR-OR-10)."""
    cp = MemoryPlatformCheckpointer()
    orig_run = "run-orig-100"

    rec1 = cp.save_checkpoint(orig_run, {"run_id": orig_run, "value": 10}, "step1")
    _ = cp.save_checkpoint(orig_run, {"run_id": orig_run, "value": 20}, "step2")

    # Fork from step 1 into a new run
    fork_run = "run-fork-200"
    forked = cp.fork_checkpoint(
        source_checkpoint_id=rec1.checkpoint_id,
        new_run_id=fork_run,
        state_overrides={"value": 999, "branch_reason": "testing alternate prompt"},
    )

    assert forked.run_id == fork_run
    assert forked.state["value"] == 999
    assert forked.state["branch_reason"] == "testing alternate prompt"
    assert forked.metadata["forked_from_checkpoint"] == rec1.checkpoint_id
    assert forked.metadata["forked_from_run"] == orig_run

    # Error handling on duplicate run or missing source
    with pytest.raises(ConflictError):
        cp.fork_checkpoint(rec1.checkpoint_id, new_run_id=fork_run)

    with pytest.raises(NotFoundError):
        cp.fork_checkpoint("chk-nonexistent", new_run_id="run-fresh")


@pytest.mark.unit
def test_sql_checkpointer_crud_and_crash_recovery(tmp_path: pytest.TempPathFactory) -> None:
    """Validate SQL checkpointer persistence across engine restarts (crash simulation)."""
    db_file = tmp_path / "checkpoints.db"  # type: ignore[operator]
    db_url = f"sqlite:///{db_file}"

    run_id = "run-crash-test"

    # --- Phase 1: Worker 1 saves checkpoint and simulated crash occurs ---
    worker1_cp = SQLPlatformCheckpointer(db_url=db_url)
    state_before_crash = {
        "run_id": run_id,
        "tenant_id": "tenant-corp",
        "step_index": 2,
        "completed_steps": ["extract", "transform"],
        "pending_step": "load_database",
    }
    saved_rec = worker1_cp.save_checkpoint(
        run_id=run_id,
        state=state_before_crash,
        step_name="transform_node",
        metadata={"worker_id": "pod-worker-77a"},
    )
    assert saved_rec.checkpoint_id.startswith("chk-")

    # Simulate worker crash: worker1 instance is disposed
    worker1_cp.engine.dispose()
    del worker1_cp

    # --- Phase 2: Replacement Worker 2 boots up and resumes run from DB ---
    worker2_cp = SQLPlatformCheckpointer(db_url=db_url)
    resumed_rec = worker2_cp.get_latest_checkpoint(run_id)

    assert resumed_rec is not None
    assert resumed_rec.checkpoint_id == saved_rec.checkpoint_id
    assert resumed_rec.step_name == "transform_node"
    assert resumed_rec.state["completed_steps"] == ["extract", "transform"]
    assert resumed_rec.state["pending_step"] == "load_database"

    # Worker 2 continues execution and writes next step
    state_resumed = dict(resumed_rec.state)
    state_resumed["completed_steps"].append("load_database")
    state_resumed["pending_step"] = None

    worker2_cp.save_checkpoint(
        run_id=run_id,
        state=state_resumed,
        step_name="load_node",
        metadata={"worker_id": "pod-worker-88b"},
    )

    history = worker2_cp.list_checkpoints(run_id)
    assert len(history) == 2
    assert history[0].step_name == "transform_node"
    assert history[1].step_name == "load_node"

    # Test SQL forking (FR-OR-10)
    forked_rec = worker2_cp.fork_checkpoint(
        source_checkpoint_id=resumed_rec.checkpoint_id,
        new_run_id="run-forked-recovery",
        state_overrides={"retry_count": 1},
    )
    assert forked_rec.run_id == "run-forked-recovery"
    assert forked_rec.state["retry_count"] == 1

    # Cleanup
    worker2_cp.delete_checkpoints(run_id)
    assert worker2_cp.get_latest_checkpoint(run_id) is None
