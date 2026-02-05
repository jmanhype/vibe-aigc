"""Test workflow resume from checkpoint functionality."""

import pytest
import tempfile
import os
import shutil
import asyncio
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus, WorkflowExecutor
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager

@pytest.mark.asyncio
class TestWorkflowResume:
    """Test workflow checkpoint and resume functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_executor_resume_from_checkpoint(self):
        """Test that WorkflowExecutor can resume from checkpoint correctly."""

        # Create a workflow with multiple nodes
        node1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First step")
        node2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE,
                            description="Second step", dependencies=["step-1"])
        node3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE,
                            description="Third step", dependencies=["step-2"])

        plan = WorkflowPlan(
            id="resume-test-plan",
            source_vibe=Vibe(description="Resume test workflow"),
            root_nodes=[node1, node2, node3]
        )

        # Create a partial execution result (step-1 completed)
        partial_result = ExecutionResult("resume-test-plan")
        partial_result.add_node_result(NodeResult("step-1", ExecutionStatus.COMPLETED,
                                                 result={"analysis": "done"}, duration=0.5))

        # Create checkpoint with partial execution
        checkpoint = WorkflowCheckpoint(plan, partial_result)

        # Resume execution from checkpoint
        executor = WorkflowExecutor()
        result = await executor.execute_plan(plan, checkpoint)

        # Verify that all nodes completed
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3
        assert result.node_results["step-1"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-2"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-3"].status == ExecutionStatus.COMPLETED

        # Verify that step-1 was not re-executed (duration should remain the same)
        assert result.node_results["step-1"].duration == 0.5
