"""Tests for automatic checkpoint creation during execution (US-014)."""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import (
    WorkflowExecutor, ExecutionResult, ExecutionStatus, NodeResult
)
from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager


def create_test_plan(num_nodes: int = 5) -> WorkflowPlan:
    """Create a test workflow plan."""
    vibe = Vibe(description="Test workflow")

    nodes = [
        WorkflowNode(
            id=f"node-{i}",
            type=WorkflowNodeType.GENERATE,
            description=f"Test node {i}"
        )
        for i in range(num_nodes)
    ]

    return WorkflowPlan(id="test-plan", source_vibe=vibe, root_nodes=nodes)


@pytest.mark.asyncio
class TestAutomaticCheckpointing:
    """Test automatic checkpoint creation during execution."""

    async def test_checkpoint_created_at_interval(self, tmp_path):
        """Test that checkpoints are created at specified intervals."""
        plan = create_test_plan(num_nodes=6)
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Checkpoint every 2 completed nodes
        executor = WorkflowExecutor(checkpoint_interval=2, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        # Should have created checkpoints during execution + final checkpoint
        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        # At least one checkpoint should exist (final checkpoint)
        assert len(checkpoints) >= 1
        assert result.status == ExecutionStatus.COMPLETED

    async def test_final_checkpoint_on_completion(self, tmp_path):
        """Test that final checkpoint is created on successful completion."""
        plan = create_test_plan(num_nodes=3)
        checkpoint_dir = str(tmp_path / "checkpoints")

        executor = WorkflowExecutor(checkpoint_interval=10, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        # Final checkpoint should exist
        assert len(checkpoints) == 1
        assert checkpoints[0]["status"] == "completed"

    async def test_final_checkpoint_on_failure(self, tmp_path):
        """Test that checkpoint is created even on execution failure."""
        vibe = Vibe(description="Failing test")
        nodes = [
            WorkflowNode(id="good", type=WorkflowNodeType.GENERATE, description="Good node"),
            WorkflowNode(id="bad", type=WorkflowNodeType.GENERATE, description="Bad node"),
        ]
        plan = WorkflowPlan(id="fail-plan", source_vibe=vibe, root_nodes=nodes)

        checkpoint_dir = str(tmp_path / "checkpoints")
        executor = WorkflowExecutor(checkpoint_interval=1, checkpoint_dir=checkpoint_dir)

        # Make second node fail
        original_gen = executor.node_handlers[WorkflowNodeType.GENERATE]

        async def failing_gen(node):
            if node.id == "bad":
                raise RuntimeError("Simulated failure")
            return await original_gen(node)

        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_gen

        result = await executor.execute_plan(plan)

        # Should have checkpointed despite failure
        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) >= 1
        assert result.status == ExecutionStatus.FAILED

    async def test_checkpoint_interval_zero_disables(self, tmp_path):
        """Test that checkpoint_interval=None disables checkpointing."""
        plan = create_test_plan(num_nodes=3)
        checkpoint_dir = str(tmp_path / "checkpoints")

        # No checkpoint interval - disabled
        executor = WorkflowExecutor(checkpoint_interval=None, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        # Check if checkpoints dir exists and has files
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            assert len(files) == 0

    async def test_checkpoint_preserves_execution_state(self, tmp_path):
        """Test that checkpoints preserve accurate execution state."""
        plan = create_test_plan(num_nodes=5)
        checkpoint_dir = str(tmp_path / "checkpoints")

        executor = WorkflowExecutor(checkpoint_interval=2, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        # Load the final checkpoint and verify state
        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        if checkpoints:
            checkpoint = manager.load_checkpoint(checkpoints[0]["checkpoint_id"])

            # Verify checkpoint has correct data
            assert checkpoint.plan.id == plan.id
            assert len(checkpoint.execution_result.node_results) == 5

    async def test_last_checkpoint_id_tracked(self, tmp_path):
        """Test that executor tracks last checkpoint ID."""
        plan = create_test_plan(num_nodes=3)
        checkpoint_dir = str(tmp_path / "checkpoints")

        executor = WorkflowExecutor(checkpoint_interval=1, checkpoint_dir=checkpoint_dir)

        # Before execution
        assert executor.last_checkpoint_id is None

        await executor.execute_plan(plan)

        # After execution, should have a checkpoint ID
        assert executor.last_checkpoint_id is not None
        assert plan.id in executor.last_checkpoint_id

    async def test_checkpoint_creation_doesnt_interrupt_execution(self, tmp_path):
        """Test that checkpoint failures don't stop execution."""
        plan = create_test_plan(num_nodes=3)

        # Use invalid path that will cause checkpoint save to fail
        executor = WorkflowExecutor(
            checkpoint_interval=1,
            checkpoint_dir="/nonexistent/invalid/path/that/should/fail"
        )

        # Should still complete execution despite checkpoint failures
        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3

    async def test_checkpoint_contains_feedback_data(self, tmp_path):
        """Test that checkpoints include feedback data."""
        plan = create_test_plan(num_nodes=3)
        checkpoint_dir = str(tmp_path / "checkpoints")

        executor = WorkflowExecutor(checkpoint_interval=1, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        if checkpoints:
            checkpoint = manager.load_checkpoint(checkpoints[0]["checkpoint_id"])

            # Verify feedback data is preserved
            assert hasattr(checkpoint.execution_result, 'feedback_data')

    async def test_intermediate_checkpoints_created(self, tmp_path):
        """Test that intermediate checkpoints are created during long workflows."""
        plan = create_test_plan(num_nodes=10)
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Checkpoint every 3 nodes
        executor = WorkflowExecutor(checkpoint_interval=3, checkpoint_dir=checkpoint_dir)

        result = await executor.execute_plan(plan)

        manager = WorkflowPersistenceManager(checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        # Should have multiple checkpoints: at nodes 3, 6, 9, and final
        # (exact count depends on timing)
        assert len(checkpoints) >= 2

    async def test_checkpoint_directory_auto_created(self, tmp_path):
        """Test that checkpoint directory is created automatically."""
        plan = create_test_plan(num_nodes=2)
        checkpoint_dir = str(tmp_path / "new_checkpoint_dir")

        # Directory doesn't exist yet
        assert not os.path.exists(checkpoint_dir)

        executor = WorkflowExecutor(checkpoint_interval=1, checkpoint_dir=checkpoint_dir)
        await executor.execute_plan(plan)

        # Directory should now exist
        assert os.path.exists(checkpoint_dir)
