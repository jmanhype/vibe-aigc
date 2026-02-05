"""Test automatic checkpoint creation during execution."""

import pytest
import tempfile
import os
import shutil
import asyncio

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus, WorkflowExecutor
from vibe_aigc.persistence import WorkflowPersistenceManager

@pytest.mark.asyncio
class TestAutomaticCheckpoints:
    """Test automatic checkpoint creation functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_automatic_checkpoint_creation_by_interval(self):
        """Test that checkpoints are automatically created at specified intervals."""

        # Create workflow with multiple nodes to trigger interval-based checkpoints
        nodes = []
        for i in range(5):
            node = WorkflowNode(
                id=f"task-{i+1}",
                type=WorkflowNodeType.GENERATE,
                description=f"Task {i+1}",
                dependencies=[f"task-{i}"] if i > 0 else []
            )
            nodes.append(node)

        plan = WorkflowPlan(
            id="auto-checkpoint-test",
            source_vibe=Vibe(description="Automatic checkpoint test"),
            root_nodes=nodes
        )

        # Execute with checkpoint interval of 2 nodes
        executor = WorkflowExecutor(
            checkpoint_interval=2,
            checkpoint_dir=self.temp_dir
        )

        result = await executor.execute_plan(plan)

        # Verify execution completed
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 5

        # Verify checkpoints were created
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        checkpoints = persistence_manager.list_checkpoints()

        # Should have at least 2 checkpoints: at interval (2,4) and final
        assert len(checkpoints) >= 2

        # Verify checkpoint content includes the plan ID
        for checkpoint_info in checkpoints:
            assert checkpoint_info["plan_id"] == "auto-checkpoint-test"

        # Verify executor tracks last checkpoint
        assert executor.last_checkpoint_id is not None

    async def test_automatic_checkpoint_disabled_by_default(self):
        """Test that checkpointing is disabled when checkpoint_interval is None."""

        node = WorkflowNode(id="single-task", type=WorkflowNodeType.ANALYZE, description="Single task")
        plan = WorkflowPlan(
            id="no-checkpoint-test",
            source_vibe=Vibe(description="No checkpoint test"),
            root_nodes=[node]
        )

        # Execute without checkpoint interval (default None)
        executor = WorkflowExecutor(checkpoint_dir=self.temp_dir)
        result = await executor.execute_plan(plan)

        # Verify execution completed
        assert result.status == ExecutionStatus.COMPLETED

        # Verify no checkpoints were created
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        checkpoints = persistence_manager.list_checkpoints()
        assert len(checkpoints) == 0

        # Verify no checkpoint ID is tracked
        assert executor.last_checkpoint_id is None

    async def test_final_checkpoint_creation_on_completion(self):
        """Test that a final checkpoint is created when workflow completes."""

        node1 = WorkflowNode(id="final-task-1", type=WorkflowNodeType.ANALYZE, description="Task 1")
        node2 = WorkflowNode(id="final-task-2", type=WorkflowNodeType.GENERATE,
                            description="Task 2", dependencies=["final-task-1"])

        plan = WorkflowPlan(
            id="final-checkpoint-test",
            source_vibe=Vibe(description="Final checkpoint test"),
            root_nodes=[node1, node2]
        )

        # Execute with large interval (so only final checkpoint is created)
        executor = WorkflowExecutor(
            checkpoint_interval=10,  # Larger than number of nodes
            checkpoint_dir=self.temp_dir
        )

        result = await executor.execute_plan(plan)

        # Verify execution completed
        assert result.status == ExecutionStatus.COMPLETED

        # Verify final checkpoint was created
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        checkpoints = persistence_manager.list_checkpoints()
        assert len(checkpoints) == 1  # Only final checkpoint

        # Load and verify the checkpoint contains complete execution
        checkpoint = persistence_manager.load_checkpoint(checkpoints[0]["checkpoint_id"])
        assert len(checkpoint.execution_result.node_results) == 2
        assert checkpoint.execution_result.status == ExecutionStatus.COMPLETED

    async def test_checkpoint_creation_on_failure(self):
        """Test that checkpoint is created even when workflow fails."""

        # Create workflow where second node will fail
        node1 = WorkflowNode(id="success-task", type=WorkflowNodeType.ANALYZE, description="Will succeed")

        # Create a node that will fail by having invalid parameters for the mock handler
        node2 = WorkflowNode(
            id="failure-task",
            type=WorkflowNodeType.GENERATE,
            description="Will fail",
            parameters={"should_fail": True},  # This will be handled by mock to simulate failure
            dependencies=["success-task"]
        )

        plan = WorkflowPlan(
            id="failure-checkpoint-test",
            source_vibe=Vibe(description="Failure checkpoint test"),
            root_nodes=[node1, node2]
        )

        # Mock the generate handler to fail on specific parameters
        executor = WorkflowExecutor(
            checkpoint_interval=1,
            checkpoint_dir=self.temp_dir
        )

        # Replace the generate handler to simulate failure
        async def failing_generate(node):
            if node.parameters.get("should_fail"):
                raise RuntimeError("Simulated task failure")
            return {"type": "generation", "description": node.description, "result": "success"}

        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_generate

        # Execute workflow (should fail)
        result = await executor.execute_plan(plan)

        # Verify execution failed
        assert result.status == ExecutionStatus.FAILED

        # Verify checkpoint was still created
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        checkpoints = persistence_manager.list_checkpoints()
        assert len(checkpoints) >= 1

        # Load checkpoint and verify it contains partial execution
        checkpoint = persistence_manager.load_checkpoint(checkpoints[0]["checkpoint_id"])

        # Should have at least the successful first task
        assert "success-task" in checkpoint.execution_result.node_results
        assert checkpoint.execution_result.node_results["success-task"].status == ExecutionStatus.COMPLETED